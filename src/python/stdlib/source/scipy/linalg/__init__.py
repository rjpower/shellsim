"""shellsim's ``scipy.linalg``.

Built over the dense real-matrix kernels shared with ``numpy.linalg`` (see
``src/python/stdlib/numpy/linalg/dense.rs``) and, for the f2py-style LAPACK and BLAS wrappers
(:mod:`scipy.linalg.lapack`, :mod:`scipy.linalg.blas`), the native ``_scipy_linalg`` module
directly. Every native routine here works on one matrix at a time, matching LAPACK's own
single-call shape; the functions below loop over stacked leading dimensions in Python where
SciPy batches them, which charges CPU for the whole stack automatically since every native call
charges its own cubic work.

Accuracy target: results agree with SciPy to about `1e-12` relative for well-conditioned input.
Bitwise agreement with SciPy's OpenBLAS is not a goal; see ``docs/scipy.md``.
"""

import math
import warnings

import numpy as np
import _numpy_linalg
import _scipy_linalg
from numpy.linalg import LinAlgError, norm  # noqa: F401  (re-exported)


class LinAlgWarning(RuntimeWarning):
    """Warned for a singular or ill-conditioned matrix, as SciPy's own does."""


__all__ = [
    "LinAlgError",
    "LinAlgWarning",
    "solve",
    "solve_triangular",
    "solve_banded",
    "solve_circulant",
    "inv",
    "det",
    "lu",
    "lu_factor",
    "lu_solve",
    "cholesky",
    "cho_factor",
    "cho_solve",
    "qr",
    "eigh",
    "eigvalsh",
    "svd",
    "svdvals",
    "diagsvd",
    "lstsq",
    "pinv",
    "pinvh",
    "null_space",
    "orth",
    "subspace_angles",
    "polar",
    "orthogonal_procrustes",
    "expm",
    "coshm",
    "sinhm",
    "tanhm",
    "khatri_rao",
    "norm",
    "bandwidth",
    "issymmetric",
    "ishermitian",
    "toeplitz",
    "circulant",
    "hankel",
    "hadamard",
    "leslie",
    "block_diag",
    "companion",
    "helmert",
    "hilbert",
    "invhilbert",
    "pascal",
    "invpascal",
    "fiedler",
    "fiedler_companion",
    "convolution_matrix",
    "dft",
]

# -------------------------------------------------------------------------------------------
# Shared helpers
# -------------------------------------------------------------------------------------------


def _resolve_precision(function, *arrays):
    """The shared working dtype for `arrays`, as real SciPy's LAPACK dispatch resolves it:
    `float32` stays `float32`; everything else (including `float16` and `bool`, with SciPy's
    own one-time `DeprecationWarning`) promotes to `float64`. Rejects complex input with
    shellsim's unsupported-feature diagnostic, matching `docs/scipy.md`'s unsupported frontier.
    """
    warn_dtype = None
    all_single = True
    for a in arrays:
        _scipy_linalg.check_real(a, function)
        if a.dtype in (np.dtype(np.float16), np.dtype(np.bool_)):
            if warn_dtype is None:
                warn_dtype = a.dtype
        elif a.dtype != np.dtype(np.float32):
            all_single = False
    if warn_dtype is not None:
        warnings.warn(
            f"Calling linalg.{function} with arguments of dtype={warn_dtype} "
            f"(a.dtype.char = '{warn_dtype.char}') is deprecated in SciPy 1.18.0 and will be "
            "removed in SciPy 1.20.0. Please cast array inputs to one of np.float{32,64} or "
            "np.complex{64,128} manually.",
            DeprecationWarning,
            stacklevel=3,
        )
        return np.dtype(np.float32)
    return np.dtype(np.float32) if all_single else np.dtype(np.float64)


def _prefix(dtype):
    return "s" if dtype == np.dtype(np.float32) else "d"


def _check_square(a, name):
    if a.ndim < 2 or a.shape[-1] != a.shape[-2]:
        raise ValueError(f"Expected square matrix, got {name}.shape={a.shape}")


def _check_finite(*arrays):
    for a in arrays:
        if not np.all(np.isfinite(a)):
            raise ValueError("array must not contain infs or NaNs")


def _symmetrize(a, lower):
    """A full symmetric matrix built from the triangle `lower` names, mirroring it into the
    other (the other triangle of `a` is not read)."""
    trusted = np.tril(a) if lower else np.triu(a)
    diagonal = np.diagonal(a, axis1=-2, axis2=-1)
    identity = np.eye(a.shape[-1], dtype=a.dtype)
    return trusted + np.swapaxes(trusted, -1, -2) - diagonal[..., None] * identity


def _lange(norm_kind, a):
    prefix = _prefix(a.dtype)
    return getattr(_scipy_linalg, f"{prefix}lange")(norm_kind.encode(), a)


def _rcond_1norm(a, inv_a):
    """The exact 1-norm reciprocal condition number `1 / (norm_1(a) * norm_1(inv_a))`.

    LAPACK's `gecon` estimates this iteratively (Hager's method) without forming the inverse.
    `inv_a` is the explicit inverse the caller already has from the same factorization (via
    `getri` or `potri`), which gives the exact value at the same cubic cost `gecon` itself
    charges internally, so this composes it directly instead of estimating it.
    """
    anorm = _lange("1", a)
    norm_inverse = _lange("1", inv_a)
    if anorm <= 0.0 or norm_inverse == 0.0 or not math.isfinite(norm_inverse):
        return 0.0
    return 1.0 / (anorm * norm_inverse)


def _warn_if_ill_conditioned(rcond, dtype, slice_index=0):
    """Warn as SciPy's `solve` does when the reciprocal condition number `rcond` (from `gecon`
    or `_rcond_1norm`) is below the dtype's machine epsilon, SciPy's own threshold."""
    if rcond < np.finfo(dtype).eps:
        warnings.warn(
            f"An ill-conditioned matrix detected: slice {slice_index} has rcond = {rcond}.",
            LinAlgWarning,
            stacklevel=4,
        )


def _map_batch(func, arrays, core_ndims):
    """Call `func(*2d_or_1d_slices)` once per leading-axis index when any array in `arrays` has
    more axes than its `core_ndims` entry, stacking the (possibly multiple) results back
    together; otherwise call `func` once directly. Batched arrays must share their leading
    axis's length. This is the shared batching strategy `docs/scipy.md` describes: SciPy's own
    batching is native, shellsim's is this Python loop over otherwise single-matrix native
    calls.
    """
    extra = [np.ndim(a) - core for a, core in zip(arrays, core_ndims)]
    if max(extra) <= 0:
        return func(*arrays)
    n = next(np.shape(a)[0] for a, e in zip(arrays, extra) if e > 0)
    results = [
        _map_batch(
            func,
            [a[i] if e > 0 else a for a, e in zip(arrays, extra)],
            core_ndims,
        )
        for i in range(n)
    ]
    if isinstance(results[0], tuple):
        return tuple(np.stack(parts) for parts in zip(*results))
    return np.stack(results)


# -------------------------------------------------------------------------------------------
# solve
# -------------------------------------------------------------------------------------------

_SYMMETRIC_STRUCTURES = {"sym", "symmetric", "her", "hermitian"}
_POSITIVE_STRUCTURES = {"pos", "positive definite"}
_STRUCTURES = _SYMMETRIC_STRUCTURES | _POSITIVE_STRUCTURES | {
    "gen", "general", "diagonal", "tridiagonal", "banded", "lower triangular", "upper triangular",
}


def solve(a, b, lower=False, overwrite_a=False, overwrite_b=False, check_finite=True,
          assume_a=None, transposed=False):
    """The solution `x` of `a @ x == b`.

    `assume_a` selects the algorithm: ``None``/``"gen"`` (general LU with partial pivoting),
    ``"sym"``/``"her"`` (treated as general after symmetrizing the triangle `lower` names, since
    shellsim has no dedicated indefinite solver), ``"pos"`` (Cholesky), or ``"diagonal"``,
    ``"tridiagonal"``, ``"lower triangular"``/``"upper triangular"`` (all solved as general LU,
    which is exact for these structured matrices too, just without their specialized LAPACK
    kernels' speed). `transposed` solves `a.T @ x == b` instead.
    """
    a = np.asarray(a)
    b = np.asarray(b)
    if assume_a is not None and assume_a not in _STRUCTURES:
        raise ValueError(f"{assume_a} is not a recognized matrix structure")
    _check_square(a, "a1")
    # `b`'s core (non-batch) solve dimension is its second-to-last axis when it carries the same
    # number of axes as `a` (a stack of right-hand-side matrices), else its last axis (a stack of
    # right-hand-side vectors, or the unbatched case).
    b_n = b.shape[-2] if b.ndim == a.ndim else b.shape[-1]
    if a.shape[-1] != b_n:
        raise ValueError(f"incompatible shapes: a1.shape={a.shape} and b1.shape={b.shape + (1,) if b.ndim == 1 else b.shape}")
    if check_finite:
        _check_finite(a, b)
    dtype = _resolve_precision("solve", a, b)
    a = a.astype(dtype)
    b = b.astype(dtype)
    prefix = _prefix(dtype)
    getrf = getattr(_scipy_linalg, f"{prefix}getrf")
    getrs = getattr(_scipy_linalg, f"{prefix}getrs")
    gecon = getattr(_scipy_linalg, f"{prefix}gecon")
    potrf = getattr(_scipy_linalg, f"{prefix}potrf")
    potrs = getattr(_scipy_linalg, f"{prefix}potrs")
    potri = getattr(_scipy_linalg, f"{prefix}potri")

    def one(A, B):
        vector = B.ndim == 1
        rhs = B[:, None] if vector else B
        if assume_a in _POSITIVE_STRUCTURES:
            c, info = potrf(A, lower=lower)
            if info != 0:
                raise LinAlgError("A singular matrix detected: slice(s) [0] are singular.")
            # No native `pocon`: compose the same exact-inverse technique `gecon` uses natively,
            # from `potri`'s explicit inverse of the Cholesky factor already computed above.
            inv_a, _ = potri(c, lower=lower)
            _warn_if_ill_conditioned(_rcond_1norm(_symmetrize(A, lower), inv_a), dtype)
            x, _ = potrs(c, rhs, lower=lower)
        else:
            if assume_a in _SYMMETRIC_STRUCTURES:
                A = _symmetrize(A, lower)
            lu, piv, info = getrf(A)
            if info != 0:
                raise LinAlgError("A singular matrix detected: slice(s) [0] are singular.")
            rcond, _ = gecon(lu, _lange("1", A))
            _warn_if_ill_conditioned(rcond, dtype)
            x, _ = getrs(lu, piv, rhs, trans=1 if transposed else 0)
        return x[:, 0] if vector else x

    # `getrs`/`potrs` return a Fortran-ordered array (matching LAPACK), but `scipy.linalg.solve`
    # itself returns a C-ordered one.
    return np.ascontiguousarray(_map_batch(one, [a, b], [2, b.ndim if b.ndim < a.ndim else 2]))


def solve_triangular(a, b, trans=0, lower=False, unit_diagonal=False, overwrite_b=False,
                      check_finite=True):
    """The solution `x` of the triangular system `a @ x == b` (or, for `trans` in
    ``(1, "T")``, `a.T @ x == b`)."""
    a = np.asarray(a)
    b = np.asarray(b)
    _check_square(a, "a1")
    if check_finite:
        _check_finite(a, b)
    dtype = _resolve_precision("solve_triangular", a, b)
    a = a.astype(dtype)
    b = b.astype(dtype)
    prefix = _prefix(dtype)
    trtrs = getattr(_scipy_linalg, f"{prefix}trtrs")
    trans_code = {0: 0, "N": 0, 1: 1, "T": 1, 2: 2, "C": 2}[trans]

    def one(A, B):
        vector = B.ndim == 1
        rhs = B[:, None] if vector else B
        x, info = trtrs(A, rhs, lower=lower, trans=trans_code, unitdiag=unit_diagonal)
        if info > 0:
            # `trtrs`'s `info` is LAPACK's own 1-based failing row; SciPy's message reports it
            # 0-based instead.
            raise LinAlgError(f"singular matrix: resolution failed at diagonal {info - 1}")
        return x[:, 0] if vector else x

    return _map_batch(one, [a, b], [2, b.ndim if b.ndim < a.ndim else 2])


def solve_banded(l_and_u, ab, b, overwrite_ab=False, overwrite_b=False, check_finite=True):
    """The solution of the banded system stored in `ab` (LAPACK's compact banded form: row
    `u - i + j` of column `j` holds `A[i, j]`, for `kl` sub- and `ku` super-diagonals)."""
    kl, ku = l_and_u
    ab = np.asarray(ab)
    b = np.asarray(b)
    if ab.shape[0] != kl + ku + 1:
        raise ValueError(
            "invalid values for the number of lower and upper diagonals: l+u+1 "
            f"({kl + ku + 1}) does not equal ab.shape[0] ({ab.shape[0]})"
        )
    if check_finite:
        _check_finite(ab, b)
    dtype = _resolve_precision("solve_banded", ab, b)
    ab = ab.astype(dtype)
    b = b.astype(dtype)
    n = ab.shape[1]
    prefix = _prefix(dtype)
    gbsv = getattr(_scipy_linalg, f"{prefix}gbsv")
    full = np.zeros((2 * kl + ku + 1, n), dtype=dtype)
    full[kl:, :] = ab
    vector = b.ndim == 1
    rhs = b[:, None] if vector else b
    _, _, x, info = gbsv(kl, ku, full, rhs)
    if info != 0:
        raise LinAlgError("A singular matrix detected: slice(s) [0] are singular.")
    return x[:, 0] if vector else x


def solve_circulant(c, b, singular="raise", tol=None, caxis=-1, baxis=0, outaxis=0):
    """The solution of the circulant system whose first column is `c`, via the FFT-diagonalized
    circulant matrix (`C = F^-1 diag(F c) F`, real for real `c` and `b`)."""
    c = np.asarray(c, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    eigenvalues = np.fft.fft(c)
    b_hat = np.fft.fft(b, axis=0)
    x_hat = b_hat / eigenvalues[:, None] if b.ndim > 1 else b_hat / eigenvalues
    return np.fft.ifft(x_hat, axis=0).real


# -------------------------------------------------------------------------------------------
# inv, det
# -------------------------------------------------------------------------------------------


def inv(a, overwrite_a=False, check_finite=True, assume_a=None):
    """The multiplicative inverse of a square matrix, or a stack of them."""
    a = np.asarray(a)
    _check_square(a, "a1")
    if check_finite:
        _check_finite(a)
    dtype = _resolve_precision("inv", a)
    a = a.astype(dtype)
    prefix = _prefix(dtype)
    getrf = getattr(_scipy_linalg, f"{prefix}getrf")
    getri = getattr(_scipy_linalg, f"{prefix}getri")
    potrf = getattr(_scipy_linalg, f"{prefix}potrf")
    potri = getattr(_scipy_linalg, f"{prefix}potri")

    def one(A):
        if A.shape[0] == 0:
            return np.zeros((0, 0), dtype=dtype)
        if assume_a in _POSITIVE_STRUCTURES:
            c, info = potrf(A, lower=False)
            if info != 0:
                raise LinAlgError("A singular matrix detected: slice(s) [0] are singular.")
            result, _ = potri(c, lower=False)
            return result
        lu, piv, info = getrf(A)
        if info != 0:
            raise LinAlgError("A singular matrix detected: slice(s) [0] are singular.")
        result, _ = getri(lu, piv)
        return result

    # The native `getri`/`potri` routines return Fortran-ordered arrays (matching LAPACK), but
    # `scipy.linalg.inv` itself returns a C-ordered one.
    return np.ascontiguousarray(_map_batch(one, [a], [2]))


def det(a, overwrite_a=False, check_finite=True):
    """The determinant of a square matrix, or a stack of them."""
    a = np.asarray(a)
    if a.ndim < 2 or a.shape[-1] != a.shape[-2]:
        raise ValueError(
            f"Last 2 dimensions of the array must be square but received shape {a.shape}."
        )
    if check_finite:
        _check_finite(a)
    dtype = _resolve_precision("det", a)
    a = a.astype(dtype)
    prefix = _prefix(dtype)
    getrf = getattr(_scipy_linalg, f"{prefix}getrf")

    def one(A):
        n = A.shape[0]
        if n == 0:
            return dtype.type(1.0)
        lu, piv, _ = getrf(A)
        sign = 1.0
        for i, p in enumerate(piv):
            if p != i:
                sign = -sign
        value = sign
        for i in range(n):
            value *= lu[i, i]
        return dtype.type(value)

    return _map_batch(one, [a], [2])


# -------------------------------------------------------------------------------------------
# lu, lu_factor, lu_solve
# -------------------------------------------------------------------------------------------


def _extract_lu(packed, m, n):
    k = min(m, n)
    l = np.tril(packed[:, :k], -1) + np.eye(m, k, dtype=packed.dtype)
    u = np.triu(packed[:k, :])
    return l, u


def lu_factor(a, overwrite_a=False, check_finite=True):
    """The packed LU factorization LAPACK's `getrf` leaves (`lu`, and 0-based pivot indices
    `piv` such that row `i` was exchanged with row `piv[i]`)."""
    a = np.asarray(a)
    _check_square(a, "a1")
    if check_finite:
        _check_finite(a)
    dtype = _resolve_precision("lu_factor", a)
    a = a.astype(dtype)
    prefix = _prefix(dtype)
    getrf = getattr(_scipy_linalg, f"{prefix}getrf")

    def one(A):
        lu, piv, info = getrf(A)
        if info > 0:
            warnings.warn(
                f"Diagonal number {info} is exactly zero. Singular matrix.",
                LinAlgWarning,
                stacklevel=3,
            )
        return lu, piv

    return _map_batch(one, [a], [2])


def lu_solve(lu_and_piv, b, trans=0, overwrite_b=False, check_finite=True):
    """The solution of `a @ x == b` (or, `trans=1`/`2`, `a.T @ x == b`) given `a`'s packed LU
    factorization from :func:`lu_factor`."""
    lu, piv = lu_and_piv
    lu = np.asarray(lu)
    piv = np.asarray(piv)
    b = np.asarray(b)
    if check_finite:
        _check_finite(lu, b)
    prefix = _prefix(lu.dtype)
    getrs = getattr(_scipy_linalg, f"{prefix}getrs")
    x, info = getrs(lu, piv, b, trans=trans)
    return x


def lu(a, permute_l=False, overwrite_a=False, check_finite=True, p_indices=False):
    """The LU factorization of a matrix, or a stack of them, with explicit pivoting.

    Returns `(p, l, u)` with `a == p @ l @ u` (the default), `(p @ l, u)` if `permute_l`, or
    `(indices, l, u)` if `p_indices` (an int32 row-permutation array with `a[indices] == l @ u`).
    """
    a = np.asarray(a)
    if check_finite:
        _check_finite(a)
    dtype = _resolve_precision("lu", a)
    a = a.astype(dtype)
    prefix = _prefix(dtype)
    getrf = getattr(_scipy_linalg, f"{prefix}getrf")

    def one(A):
        m, n = A.shape
        packed, piv, _ = getrf(A)
        l, u = _extract_lu(packed, m, n)
        perm = np.arange(m)
        for i, p in enumerate(piv):
            perm[i], perm[p] = perm[p], perm[i]
        indices = np.argsort(perm)
        if permute_l:
            return l[indices], u
        if p_indices:
            return indices.astype(np.int32), l, u
        p_matrix = np.zeros((m, m), dtype=dtype)
        p_matrix[perm, np.arange(m)] = 1.0
        return p_matrix, l, u

    return _map_batch(one, [a], [2])


# -------------------------------------------------------------------------------------------
# cholesky, cho_factor, cho_solve
# -------------------------------------------------------------------------------------------


def cholesky(a, lower=False, overwrite_a=False, check_finite=True):
    """The Cholesky factor of a Hermitian positive-definite matrix, or a stack of them: the
    lower-triangular `L` with `L @ L.T == a` if `lower`, else the upper-triangular `U` with
    `U.T @ U == a`."""
    a = np.asarray(a)
    if a.ndim < 2 or a.shape[-1] != a.shape[-2]:
        raise ValueError(f"Expected a square matrix or batch thereof, got a1.shape={a.shape}")
    if check_finite:
        _check_finite(a)
    dtype = _resolve_precision("cholesky", a)
    a = a.astype(dtype)
    prefix = _prefix(dtype)
    potrf = getattr(_scipy_linalg, f"{prefix}potrf")

    def one(A):
        c, info = potrf(A, lower=lower)
        if info != 0:
            raise LinAlgError(f"Internal potrf return info = [{info}] for slices [0].")
        return c

    # `potrf` returns a Fortran-ordered array (matching LAPACK), but `scipy.linalg.cholesky`
    # itself returns a C-ordered one.
    return np.ascontiguousarray(_map_batch(one, [a], [2]))


def cho_factor(a, lower=False, overwrite_a=False, check_finite=True):
    """The Cholesky factor packed as :func:`cho_solve` expects: `(c, lower)`."""
    c = cholesky(a, lower=lower, overwrite_a=overwrite_a, check_finite=check_finite)
    return c, np.array(lower)


def cho_solve(c_and_lower, b, overwrite_b=False, check_finite=True):
    """The solution of `a @ x == b` given `a`'s Cholesky factor from :func:`cho_factor`."""
    c, lower = c_and_lower
    c = np.asarray(c)
    b = np.asarray(b)
    if check_finite:
        _check_finite(c, b)
    prefix = _prefix(c.dtype)
    potrs = getattr(_scipy_linalg, f"{prefix}potrs")
    x, info = potrs(c, b, lower=bool(lower))
    return x


# -------------------------------------------------------------------------------------------
# qr
# -------------------------------------------------------------------------------------------

_QR_MODES = {"full": "complete", "qr": "complete", "economic": "reduced", "r": "r", "raw": "raw"}


def qr(a, overwrite_a=False, lwork=None, mode="full", pivoting=False, check_finite=True):
    """The QR factorization of a matrix, or a stack of them.

    `mode` is one of ``"full"``/``"qr"`` (the default: `q` is square, `r` is the full
    `rows x cols` shape), ``"economic"`` (`q` is `rows x k`, `r` is `k x cols`,
    `k = min(rows, cols)`), ``"r"`` (`r` alone, as a one-element tuple), or ``"raw"`` (LAPACK's
    packed reflectors and their scale factors, `(h, tau)`, alongside `r`). `pivoting` adds
    column pivoting (Businger and Golub 1965; see `dense::householder_qr_pivoted`), appending a
    0-based column permutation `jpvt` such that `a[:, jpvt] == q @ r`.
    """
    if mode not in _QR_MODES:
        raise ValueError("Mode argument should be one of ['full', 'qr', 'r', 'raw', 'economic']")
    a = np.asarray(a)
    if check_finite:
        _check_finite(a)
    dtype = _resolve_precision("qr", a)
    a = a.astype(dtype)

    if pivoting:
        def one(A):
            q, r, jpvt = _scipy_linalg.qr_pivoted(A)
            if mode == "economic":
                k = min(A.shape)
                q, r = q[:, :k], r[:k, :]
            if mode == "r":
                return r, jpvt
            return q, r, jpvt

        return _map_batch(one, [a], [2])

    if mode == "raw":
        h, tau = _numpy_linalg.qr(a, "raw")
        (r,) = _numpy_linalg.qr(a, "r")
        return (h, tau), r
    # `_numpy_linalg.qr(a, "r")` already returns a 1-tuple `(r,)` (matching numpy's own
    # `mode="r"` convention), so this passes it straight through rather than wrapping it again.
    return _numpy_linalg.qr(a, _QR_MODES[mode])


# -------------------------------------------------------------------------------------------
# eigh, eigvalsh
# -------------------------------------------------------------------------------------------


def eigh(a, b=None, lower=True, eigvals_only=False, overwrite_a=False, overwrite_b=False,
         turbo=True, eigvals=None, type=1, check_finite=True, subset_by_index=None,
         subset_by_value=None, driver=None):
    """Eigenvalues (ascending) and, unless `eigvals_only`, orthonormal eigenvectors of a
    Hermitian matrix `a`, or the generalized problem `a @ x == w * b @ x` when `b` is given.

    `subset_by_index` (an inclusive `[low, high]` pair of 0-based ranks) or `subset_by_value`
    (an inclusive `(low, high)` value range) keep only some eigenpairs, computed by filtering the
    full solution (shellsim's Jacobi eigensolver, unlike LAPACK's `syevr`, does not compute a
    requested subset any more cheaply than the full spectrum).
    """
    a = np.asarray(a)
    if a.ndim < 2 or a.shape[-1] != a.shape[-2]:
        raise ValueError('expected square "a" matrix')
    if check_finite:
        _check_finite(a)
    _scipy_linalg.check_real(a, "eigh")
    dtype = _resolve_precision("eigh", a)
    a = a.astype(dtype)
    if b is not None:
        b = np.asarray(b).astype(dtype)
        if check_finite:
            _check_finite(b)
        result = _scipy_linalg.eigh_gen(a, b, lower, True)
        w, v = result
    else:
        w, v = _numpy_linalg.eigh(a, "L" if lower else "U", True)
    n = w.shape[-1]
    low, high = 0, n - 1
    if subset_by_index is not None:
        low, high = subset_by_index
        if not (0 <= low <= high <= n - 1):
            raise ValueError(
                "Requested eigenvalue indices are not valid. Valid range is "
                f"[0, {n - 1}] and start <= end, but start={low}, end={high} is given"
            )
    keep = np.ones(n, dtype=bool)
    keep[:low] = False
    keep[high + 1:] = False
    if subset_by_value is not None:
        value_low, value_high = subset_by_value
        keep &= (w > value_low) & (w < value_high)
    w = w[keep]
    if eigvals_only:
        return w
    # LAPACK's `syevd`/`sygvd` overwrite their input in place and return the eigenvectors in
    # that same (Fortran-ordered) storage; match that layout, since `v[:, keep]` above always
    # makes a fresh C-ordered copy regardless of `_numpy_linalg.eigh`'s own output order.
    v = np.asfortranarray(v[:, keep])
    return w, v


def eigvalsh(a, b=None, lower=True, overwrite_a=False, overwrite_b=False, turbo=True,
             eigvals=None, type=1, check_finite=True, subset_by_index=None,
             subset_by_value=None, driver=None):
    """Eigenvalues (ascending) of a Hermitian matrix, or the generalized problem with `b`,
    without eigenvectors. See :func:`eigh`."""
    return eigh(
        a, b, lower=lower, eigvals_only=True, check_finite=check_finite,
        subset_by_index=subset_by_index, subset_by_value=subset_by_value,
    )


# -------------------------------------------------------------------------------------------
# svd, svdvals, diagsvd
# -------------------------------------------------------------------------------------------


def svd(a, full_matrices=True, compute_uv=True, overwrite_a=False, check_finite=True,
        lapack_driver="gesdd"):
    """The singular value decomposition of a matrix, or a stack of them."""
    if lapack_driver not in ("gesdd", "gesvd"):
        raise ValueError(f'lapack_driver must be "gesdd" or "gesvd", not "{lapack_driver}"')
    a = np.asarray(a)
    if check_finite:
        _check_finite(a)
    _scipy_linalg.check_real(a, "svd")
    dtype = _resolve_precision("svd", a)
    a = a.astype(dtype)
    return _numpy_linalg.svd(a, full_matrices, compute_uv)


def svdvals(a, overwrite_a=False, check_finite=True):
    """The singular values of a matrix, or a stack of them, in descending order."""
    return svd(a, compute_uv=False, check_finite=check_finite)


def diagsvd(s, m, n):
    """The `m x n` matrix with `s` on its leading diagonal and zero elsewhere."""
    s = np.asarray(s)
    result = np.zeros((m, n), dtype=s.dtype)
    k = min(m, n, s.shape[0])
    result[np.arange(k), np.arange(k)] = s[:k]
    return result


# -------------------------------------------------------------------------------------------
# lstsq, pinv, pinvh
# -------------------------------------------------------------------------------------------


def lstsq(a, b, cond=None, overwrite_a=False, overwrite_b=False, check_finite=True,
          lapack_driver=None):
    """The least-squares solution of `a @ x == b`, via `a`'s SVD.

    Returns `(x, residues, rank, singular_values)`. `residues` is the sum of squared residuals
    (per right-hand-side column, if `b` is 2-D) when `a` has more rows than columns and full
    column rank; a NaN scalar when `a` has fewer rows than columns (an underdetermined system, as
    SciPy reports it); otherwise a shape-`(0,)` array. `singular_values` is `None` for the
    `"gelsy"` driver, which does not compute them (SciPy's does not either).
    """
    driver = lapack_driver or "gelsd"
    if driver not in ("gelsd", "gelsy", "gelss"):
        raise ValueError(f'LAPACK driver "{driver}" is not found')
    a = np.asarray(a)
    b = np.asarray(b)
    if a.shape[0] != b.shape[0]:
        raise ValueError(
            "Shape mismatch: a and b should have the same number of rows "
            f"({a.shape[0]} != {b.shape[0]})."
        )
    if check_finite:
        _check_finite(a, b)
    dtype = _resolve_precision("lstsq", a, b)
    a = a.astype(dtype)
    b = b.astype(dtype)
    m, n = a.shape
    is_1d = b.ndim == 1
    b2 = b[:, None] if is_1d else b
    u, s, vt = _numpy_linalg.svd(a, False, True)
    eps = np.finfo(dtype).eps
    threshold = (cond if cond and cond > 0 else eps) * (s[0] if s.size else 0.0)
    large = s > threshold
    rank = np.sum(large)
    safe = np.where(large, s, 1.0)
    s_inv = np.where(large, 1.0 / safe, 0.0)
    utb = np.swapaxes(u, -1, -2) @ b2
    x = np.swapaxes(vt, -1, -2) @ (s_inv[:, None] * utb)
    singular_values = None if driver == "gelsy" else s
    # LAPACK's `gelsy` never reports residuals; `gelsd`/`gelss` report the sum of squared
    # residuals, reduced fully for a 1-D `b`, only for a full-column-rank overdetermined system
    # (`m > n and rank == n`); an overdetermined but rank-deficient system reports NaN, and an
    # underdetermined or square one reports an empty array, as SciPy's own does.
    if driver == "gelsy":
        residues = np.empty(0, dtype=dtype)
    elif m > n and rank == n:
        sq = np.sum((b2 - a @ x) ** 2, axis=0)
        residues = sq[0] if is_1d else sq
    elif m > n:
        residues = dtype.type(np.nan)
    else:
        residues = np.empty(0, dtype=dtype)
    if is_1d:
        x = x[:, 0]
    return x, residues, rank, singular_values


def pinv(a, atol=None, rtol=None, return_rank=False, check_finite=True):
    """The Moore-Penrose pseudo-inverse of `a`, via its SVD."""
    a = np.asarray(a)
    if check_finite:
        _check_finite(a)
    dtype = _resolve_precision("pinv", a)
    a = a.astype(dtype)
    u, s, vt = _numpy_linalg.svd(a, False, True)
    threshold = (atol or 0.0) + (rtol if rtol is not None else max(a.shape) * np.finfo(dtype).eps) * (
        s[0] if s.size else 0.0
    )
    large = s > threshold
    rank = int(np.sum(large))
    safe = np.where(large, s, 1.0)
    s_inv = np.where(large, 1.0 / safe, 0.0)
    v = np.swapaxes(vt, -1, -2)
    ut = np.swapaxes(u, -1, -2)
    result = (v * s_inv[..., None, :]) @ ut
    return (result, rank) if return_rank else result


def pinvh(a, atol=None, rtol=None, lower=True, return_rank=False, check_finite=True):
    """The Moore-Penrose pseudo-inverse of a Hermitian matrix `a`, via its eigendecomposition."""
    a = np.asarray(a)
    if check_finite:
        _check_finite(a)
    dtype = _resolve_precision("pinvh", a)
    a = a.astype(dtype)
    w, v = _numpy_linalg.eigh(a, "L" if lower else "U", True)
    s = np.abs(w)
    threshold = (atol or 0.0) + (rtol if rtol is not None else a.shape[-1] * np.finfo(dtype).eps) * (
        s.max() if s.size else 0.0
    )
    large = s > threshold
    rank = int(np.sum(large))
    safe = np.where(large, w, 1.0)
    w_inv = np.where(large, 1.0 / safe, 0.0)
    result = (v * w_inv) @ np.swapaxes(v, -1, -2)
    return (result, rank) if return_rank else result


# -------------------------------------------------------------------------------------------
# Subspaces, polar decomposition, orthogonal Procrustes
# -------------------------------------------------------------------------------------------


def null_space(a, rcond=None):
    """An orthonormal basis for the null space of `a`, from its SVD's trailing right singular
    vectors (those paired with a singular value at or below the rank tolerance)."""
    a = np.asarray(a, dtype=np.float64)
    m, n = a.shape
    u, s, vt = _numpy_linalg.svd(a, True, True)
    tol = (rcond if rcond is not None else max(m, n) * np.finfo(s.dtype).eps) * (s[0] if s.size else 0.0)
    rank = int(np.sum(s > tol))
    return np.swapaxes(vt[rank:], -1, -2)


def orth(a, rcond=None):
    """An orthonormal basis for the range (column space) of `a`, from its SVD's leading left
    singular vectors (those paired with a singular value above the rank tolerance)."""
    a = np.asarray(a, dtype=np.float64)
    m, n = a.shape
    u, s, vt = _numpy_linalg.svd(a, True, True)
    tol = (rcond if rcond is not None else max(m, n) * np.finfo(s.dtype).eps) * (s[0] if s.size else 0.0)
    rank = int(np.sum(s > tol))
    return u[:, :rank]


def subspace_angles(a, b):
    """The principal angles (radians, ascending) between the column spaces of `a` and `b`, via
    the singular values of `orth(a).T @ orth(b)`."""
    qa = orth(a)
    qb = orth(b)
    s = _numpy_linalg.svd(np.swapaxes(qa, -1, -2) @ qb, True, False)
    return np.arccos(np.clip(s, -1.0, 1.0))[::-1]


def polar(a, side="right"):
    """The polar decomposition `a = u @ p` (`side="right"`) or `a = p @ u` (`side="left"`),
    `u` orthonormal and `p` Hermitian positive semi-definite, via `a`'s SVD."""
    a = np.asarray(a, dtype=np.float64)
    u_svd, s, vt = _numpy_linalg.svd(a, False, True)
    u = u_svd @ vt
    if side == "right":
        p = np.swapaxes(vt, -1, -2) * s @ vt
    else:
        p = u_svd * s @ np.swapaxes(u_svd, -1, -2)
    return u, p


def orthogonal_procrustes(a, b, check_finite=True):
    """The orthogonal matrix `r` minimizing `norm(a @ r - b, "fro")`, and the sum of the
    singular values of `a.T @ b`, via the SVD of `a.T @ b` (Schonemann 1966)."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if check_finite:
        _check_finite(a, b)
    u, s, vt = _numpy_linalg.svd(np.swapaxes(a, -1, -2) @ b, True, True)
    r = u @ vt
    scale = float(np.sum(s))
    return r, scale


# -------------------------------------------------------------------------------------------
# Matrix functions
# -------------------------------------------------------------------------------------------


def expm(a):
    """The matrix exponential of `a`, or a stack of them (scaling and squaring with a fixed-
    order Padé approximant; see `dense::expm` and Higham 2005)."""
    a = np.asarray(a)
    dtype = _resolve_precision("expm", a)
    a = a.astype(dtype)
    return _map_batch(_scipy_linalg.expm, [a], [2])


def coshm(a):
    """`cosh` of the matrix `a`, via `(expm(a) + expm(-a)) / 2`."""
    a = np.asarray(a, dtype=np.float64)
    return (expm(a) + expm(-a)) / 2.0


def sinhm(a):
    """`sinh` of the matrix `a`, via `(expm(a) - expm(-a)) / 2`."""
    a = np.asarray(a, dtype=np.float64)
    return (expm(a) - expm(-a)) / 2.0


def tanhm(a):
    """`tanh` of the matrix `a`, via `inv(coshm(a)) @ sinhm(a)`."""
    a = np.asarray(a, dtype=np.float64)
    return np.linalg.solve(coshm(a), sinhm(a))


def khatri_rao(a, b):
    """The column-wise Khatri-Rao (matching-column Kronecker) product of `a` and `b`."""
    a = np.asarray(a)
    b = np.asarray(b)
    return (a[:, None, :] * b[None, :, :]).reshape(a.shape[0] * b.shape[0], a.shape[1])


# -------------------------------------------------------------------------------------------
# Norms and structure checks
# -------------------------------------------------------------------------------------------


def bandwidth(a):
    """The number of nonzero sub- and super-diagonals of `a`, as a `(lower, upper)` pair."""
    a = np.asarray(a)
    rows, cols = np.nonzero(a)
    if rows.size == 0:
        return 0, 0
    lower = max(0, int(np.max(rows - cols)))
    upper = max(0, int(np.max(cols - rows)))
    return lower, upper


def issymmetric(a, atol=None, rtol=None):
    """Whether `a` equals its own transpose, exactly unless `atol`/`rtol` give a tolerance."""
    a = np.asarray(a)
    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise ValueError("Input array must be square.")
    if atol is None and rtol is None:
        return bool(np.array_equal(a, np.swapaxes(a, -1, -2)))
    return bool(np.allclose(a, np.swapaxes(a, -1, -2), rtol=rtol or 0.0, atol=atol or 0.0))


def ishermitian(a, atol=None, rtol=None):
    """Whether `a` equals its own (conjugate) transpose; real `a` makes this the same as
    :func:`issymmetric`."""
    return issymmetric(a, atol=atol, rtol=rtol)


# -------------------------------------------------------------------------------------------
# Special matrices
# -------------------------------------------------------------------------------------------


def toeplitz(c, r=None):
    """The Toeplitz matrix with first column `c` and first row `r` (`c` with `r=None`, so a
    symmetric or Hermitian-shaped matrix)."""
    c = np.asarray(c).ravel()
    r = np.asarray(c).ravel() if r is None else np.asarray(r).ravel()
    n, m = len(c), len(r)
    dtype = np.result_type(c, r)
    result = np.empty((n, m), dtype=dtype)
    for i in range(n):
        for j in range(m):
            result[i, j] = c[i - j] if i >= j else r[j - i]
    return result


def circulant(c):
    """The circulant matrix whose first column is `c`; row `i` is `c` rotated by `i`."""
    c = np.asarray(c).ravel()
    n = len(c)
    result = np.empty((n, n), dtype=c.dtype)
    for i in range(n):
        for j in range(n):
            result[i, j] = c[(i - j) % n]
    return result


def hankel(c, r=None):
    """The Hankel matrix with first column `c` and last row `r` (`r`'s first entry, if given,
    must equal `c`'s last; default `r` is zero after `c[-1]`)."""
    c = np.asarray(c).ravel()
    n = len(c)
    if r is None:
        r = np.zeros_like(c)
        r[0] = c[-1]
    else:
        r = np.asarray(r).ravel()
    m = len(r)
    dtype = np.result_type(c, r)
    result = np.empty((n, m), dtype=dtype)
    for i in range(n):
        for j in range(m):
            k = i + j
            result[i, j] = c[k] if k < n else r[k - n + 1]
    return result


def _kron(a, b):
    return (a[:, None, :, None] * b[None, :, None, :]).reshape(
        a.shape[0] * b.shape[0], a.shape[1] * b.shape[1]
    )


def hadamard(n, dtype=int):
    """The `n x n` Hadamard matrix, `n` a power of 2, built by repeated Kronecker product with
    `[[1, 1], [1, -1]]`."""
    if n < 1 or (n & (n - 1)) != 0:
        raise ValueError("n must be a positive integer, and n must be a power of 2")
    base = np.array([[1, 1], [1, -1]], dtype=dtype)
    result = np.array([[1]], dtype=dtype)
    while result.shape[0] < n:
        result = _kron(base, result)
    return result


def leslie(f, s):
    """The Leslie (age-structured population growth) matrix with fecundities `f` on its first
    row and survival fractions `s` on its subdiagonal."""
    f = np.asarray(f, dtype=np.float64)
    s = np.asarray(s, dtype=np.float64)
    n = f.shape[-1]
    result = np.zeros(f.shape[:-1] + (n, n), dtype=np.float64)
    result[..., 0, :] = f
    idx = np.arange(n - 1)
    result[..., idx + 1, idx] = s
    return result


def block_diag(*arrays):
    """A block-diagonal matrix with `arrays` (each promoted to at least 2-D) along the
    diagonal, zero elsewhere."""
    if not arrays:
        return np.zeros((1, 0))
    mats = [np.atleast_2d(np.asarray(a)) for a in arrays]
    dtype = np.result_type(*[m.dtype for m in mats])
    rows = sum(m.shape[0] for m in mats)
    cols = sum(m.shape[1] for m in mats)
    result = np.zeros((rows, cols), dtype=dtype)
    r = c = 0
    for m in mats:
        result[r:r + m.shape[0], c:c + m.shape[1]] = m
        r += m.shape[0]
        c += m.shape[1]
    return result


def companion(a):
    """The companion matrix of the polynomial with coefficients `a` (leading first, and
    nonzero)."""
    a = np.asarray(a, dtype=np.float64)
    n = a.shape[-1] - 1
    if n < 1:
        raise ValueError("The length of `a` must be at least 2.")
    if a[0] == 0:
        raise ValueError(
            "The first coefficient(s) of `a` (i.e. elements of `a[..., 0]`) must not be zero."
        )
    result = np.zeros((n, n), dtype=np.float64)
    result[0, :] = -a[1:] / a[0]
    if n > 1:
        result[1:, :-1] = np.eye(n - 1)
    return result


def fiedler_companion(a):
    """The Fiedler companion matrix of the polynomial with coefficients `a` (leading first).

    Built from the ordinary companion matrix (:func:`companion`) by swapping its top-right
    entry (the lowest-degree coefficient) with its bottom sub-diagonal entry, matching M.
    Fiedler's 2003 construction for a cubic polynomial; this module does not implement
    Fiedler's general interleaving construction for higher degree.
    """
    c = companion(a)
    n = c.shape[0]
    if n >= 2:
        c[0, -1], c[-1, -2] = c[-1, -2], c[0, -1]
    return c


def helmert(n, full=False):
    """The Helmert matrix: an orthonormal contrast basis. Row 0 (included only if `full`) is
    the constant `1/sqrt(n)`; row `i` (`i >= 1`) is `1/sqrt(i(i+1))` for its first `i` entries,
    `-i/sqrt(i(i+1))` at position `i`, and zero after."""
    h = np.zeros((n, n), dtype=np.float64)
    h[0] = 1.0 / np.sqrt(n)
    for i in range(1, n):
        h[i, :i] = 1.0 / np.sqrt(i * (i + 1))
        h[i, i] = -i / np.sqrt(i * (i + 1))
    return h if full else h[1:]


def hilbert(n):
    """The `n x n` Hilbert matrix: entry `(i, j)` is `1 / (i + j + 1)`, 0-based."""
    i = np.arange(1, n + 1, dtype=np.float64)
    return 1.0 / (i[:, None] + i[None, :] - 1.0)


def invhilbert(n, exact=False):
    """The exact inverse of the `n x n` Hilbert matrix, from its closed-form integer entries
    (Wikipedia, "Hilbert matrix"), as `int64` if `exact` else rounded to `float64`."""
    result = np.empty((n, n), dtype=np.int64)
    for i in range(1, n + 1):
        for j in range(1, n + 1):
            result[i - 1, j - 1] = (
                (-1) ** (i + j)
                * (i + j - 1)
                * math.comb(n + i - 1, n - j)
                * math.comb(n + j - 1, n - i)
                * math.comb(i + j - 2, i - 1) ** 2
            )
    return result if exact else result.astype(np.float64)


def pascal(n, kind="symmetric", exact=True):
    """The `n x n` Pascal matrix of binomial coefficients: `"symmetric"`, `binom(i+j, i)`;
    `"lower"`, the triangular binomial matrix `binom(i, j)`; or `"upper"`, its transpose."""
    lower = np.zeros((n, n), dtype=np.uint64)
    for i in range(n):
        for j in range(i + 1):
            lower[i, j] = math.comb(i, j)
    if kind == "lower":
        return lower
    if kind == "upper":
        return np.swapaxes(lower, -1, -2).copy()
    return lower @ np.swapaxes(lower, -1, -2)


def invpascal(n, kind="symmetric", exact=True):
    """The exact inverse of :func:`pascal`, from the inverse of the triangular binomial
    matrix, `inv_lower[i, j] = (-1)**(i - j) * binom(i, j)`."""
    inv_lower = np.zeros((n, n), dtype=np.int64)
    for i in range(n):
        for j in range(i + 1):
            inv_lower[i, j] = (-1) ** (i - j) * math.comb(i, j)
    if kind == "lower":
        return inv_lower
    if kind == "upper":
        return np.swapaxes(inv_lower, -1, -2).copy()
    return np.swapaxes(inv_lower, -1, -2) @ inv_lower


def fiedler(a):
    """The Fiedler matrix: entry `(i, j)` is `abs(a[i] - a[j])`."""
    a = np.asarray(a)
    return np.abs(a[:, None] - a[None, :])


def convolution_matrix(a, n, mode="full"):
    """The Toeplitz matrix `c` such that `c @ x` convolves the length-`len(a)` kernel `a`
    with a length-`n` signal `x`, in the given `mode` (`"full"`, `"valid"`, or `"same"`,
    matching :func:`numpy.convolve`)."""
    a = np.asarray(a)
    m = len(a)
    full = np.zeros((m + n - 1, n), dtype=a.dtype)
    for j in range(n):
        full[j:j + m, j] = a
    if mode == "full":
        return full
    if mode == "valid":
        start = min(m, n) - 1
        count = max(m, n) - min(m, n) + 1
        return full[start:start + count]
    if mode == "same":
        start = (min(m, n) - 1) // 2
        return full[start:start + max(m, n)]
    raise ValueError(f"'mode' argument must be one of ('full', 'valid', 'same'), got '{mode}'")


def dft(n, scale=None):
    """The `n x n` Discrete Fourier Transform matrix: entry `(i, j)` is `exp(-2 pi i j k / n)`.

    `scale` is `None` (unscaled), `"sqrtn"` (unitary, divided by `sqrt(n)`), or `"n"` (divided
    by `n`, inverting an unscaled forward transform).
    """
    k = np.arange(n)
    m = np.exp(-2j * np.pi * np.outer(k, k) / n)
    if scale == "sqrtn":
        m = m / np.sqrt(n)
    elif scale == "n":
        m = m / n
    return m


# -------------------------------------------------------------------------------------------
# Unsupported frontier
# -------------------------------------------------------------------------------------------

_UNSUPPORTED_NAMES = {
    "eig", "eigvals", "eig_banded", "eigvals_banded", "eigh_tridiagonal",
    "eigvalsh_tridiagonal", "schur", "rsf2csf", "hessenberg", "cdf2rdf", "logm", "sqrtm",
    "funm", "expm_frechet", "expm_cond", "matrix_balance", "solveh_banded", "solve_lyapunov",
    "solve_sylvester", "solve_continuous_are", "solve_discrete_are",
    "solve_continuous_lyapunov", "solve_discrete_lyapunov", "cossin", "qz", "ordqz",
    "cholesky_banded", "cho_solve_banded", "matmul_toeplitz",
}


def __getattr__(name):
    if name in _UNSUPPORTED_NAMES:
        raise NotImplementedError(f"scipy.linalg.{name} is not supported by shellsim's SciPy")
    raise AttributeError(f"module 'scipy.linalg' has no attribute '{name}'")
