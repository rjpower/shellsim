"""shellsim's ``scipy.linalg``.

A small, commonly used subset of real SciPy's `scipy.linalg`, built entirely in Python over
``_numpy_linalg``'s native primitives (see ``src/python/stdlib/numpy/linalg.rs`` and
``linalg/dense.rs``): there is no native ``_scipy_linalg`` module, and no ``scipy.linalg.lapack``
or ``scipy.linalg.blas`` f2py-style wrappers. Batching over a stack of matrices comes for free
wherever it composes directly from a batched `_numpy_linalg` call (`solve`, `inv`, `det`,
`cholesky`, `eigh`, `svd`, `expm`); a few functions that need their own row-permutation or
band-storage bookkeeping (`lu`, `solve_banded`) support only a single matrix, or a single stack
of matrices, not arbitrary batch shapes.

Kept: `solve` (general, symmetric, or positive-definite), `inv`, `det`, `lu`/`lu_factor`/
`lu_solve`, `cholesky`/`cho_factor`/`cho_solve`, `qr`, `svd`/`svdvals`, `eig`/`eigvals`,
`eigh`/`eigvalsh` (including the generalized problem), `lstsq`, `pinv`, `expm`,
`solve_triangular`, `solve_banded`, `null_space`, `orth`, and the special matrices `toeplitz`,
`circulant`, `block_diag`, `hilbert`. Dropped: batched `lu`/`solve_banded` beyond a single stack,
column-pivoted QR, `coshm`/`sinhm`/`tanhm`, `polar`, `orthogonal_procrustes`,
`subspace_angles`, `pinvh`, `solve_circulant`, `diagsvd`, `khatri_rao`, the structure checks
(`bandwidth`, `issymmetric`, `ishermitian`), the exotic special matrices (`hankel`, `hadamard`,
`leslie`, `companion`, `helmert`, `invhilbert`, `pascal`, `invpascal`, `fiedler*`,
`convolution_matrix`, `dft`), and, as noted above, `scipy.linalg.lapack`/`.blas`. A missing name
is an ordinary `AttributeError`; an unsupported keyword is a `ValueError` or `NotImplementedError`.

Complex input is not supported (a short `NotImplementedError`), matching real SciPy's own complex
(`c`/`z`) LAPACK routines being outside this module's scope. Accuracy target: results agree with
SciPy to about `1e-12` relative for well-conditioned input; SciPy's OpenBLAS is not reproduced
bit-for-bit (see `docs/scipy.md`).
"""

import warnings

import numpy as np
import _numpy_linalg
from numpy.linalg import LinAlgError, norm  # noqa: F401  (re-exported)


class LinAlgWarning(RuntimeWarning):
    """Warned by :func:`lu_factor` (and, through it, :func:`lu`) when the factorization finds an
    exactly singular matrix: the factorization itself still succeeds, only a later solve would
    fail. Unlike real SciPy, shellsim does not warn for a merely ill-conditioned (but nonsingular)
    matrix in :func:`solve` or :func:`inv`, since that would need a condition-number estimate this
    module does not otherwise compute.
    """


__all__ = [
    "LinAlgError",
    "LinAlgWarning",
    "solve",
    "solve_triangular",
    "solve_banded",
    "inv",
    "det",
    "lu",
    "lu_factor",
    "lu_solve",
    "cholesky",
    "cho_factor",
    "cho_solve",
    "qr",
    "eig",
    "eigvals",
    "eigh",
    "eigvalsh",
    "svd",
    "svdvals",
    "lstsq",
    "pinv",
    "expm",
    "norm",
    "null_space",
    "orth",
    "toeplitz",
    "circulant",
    "block_diag",
    "hilbert",
]

# -------------------------------------------------------------------------------------------
# Shared helpers
# -------------------------------------------------------------------------------------------

_STRUCTURES = ("gen", "sym", "pos")
_SYMMETRIC_STRUCTURES = ("sym",)
_POSITIVE_STRUCTURES = ("pos",)


def _resolve_precision(function, *arrays):
    """The shared working dtype for `arrays`: `float32` stays `float32`; everything else
    (including `float16` and `bool`, with SciPy's own one-time `DeprecationWarning`) promotes to
    `float64`. Complex input is outside this module's scope (see the module docstring).
    """
    warn_dtype = None
    all_single = True
    for a in arrays:
        if a.dtype.kind == "c":
            raise NotImplementedError(
                f"complex input to scipy.linalg.{function} is not supported by shellsim's SciPy"
            )
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


def _broadcast_identity(a, dtype):
    """An identity matrix broadcast to `a`'s full (possibly batched) shape, as a fresh array
    (not a read-only view), for use as a right-hand side matching `a`'s batch shape exactly so
    `_numpy_linalg.solve_triangular`'s gufunc-style broadcasting always takes its "batch of
    matrices" path rather than mistaking a smaller identity for a batch of vectors.
    """
    return np.broadcast_to(np.eye(a.shape[-1], dtype=dtype), a.shape).copy()


# -------------------------------------------------------------------------------------------
# solve, solve_triangular, solve_banded
# -------------------------------------------------------------------------------------------


def solve(a, b, lower=False, overwrite_a=False, overwrite_b=False, check_finite=True,
          assume_a=None, transposed=False):
    """The solution `x` of `a @ x == b`, or a batch of them.

    `assume_a` selects the algorithm: `None`/`"gen"` (general LU with partial pivoting), `"sym"`
    (symmetrized from the triangle `lower` names, then solved as general, since shellsim has no
    dedicated indefinite solver), or `"pos"` (Cholesky, from the triangle `lower` names).
    `transposed` solves `a.T @ x == b` instead.
    """
    a = np.asarray(a)
    b = np.asarray(b)
    if assume_a is not None and assume_a not in _STRUCTURES:
        raise ValueError(f"{assume_a} is not a recognized matrix structure")
    _check_square(a, "a1")
    b_n = b.shape[-2] if b.ndim == a.ndim else b.shape[-1]
    if a.shape[-1] != b_n:
        raise ValueError(
            f"incompatible shapes: a1.shape={a.shape} and "
            f"b1.shape={b.shape + (1,) if b.ndim == 1 else b.shape}"
        )
    if check_finite:
        _check_finite(a, b)
    dtype = _resolve_precision("solve", a, b)
    a = a.astype(dtype)
    b = b.astype(dtype)
    if assume_a in _POSITIVE_STRUCTURES:
        c = _numpy_linalg.cholesky(_symmetrize(a, lower))
        y = _numpy_linalg.solve_triangular(c, b, True, False, False)
        return _numpy_linalg.solve_triangular(c, y, True, True, False)
    if assume_a in _SYMMETRIC_STRUCTURES:
        a = _symmetrize(a, lower)
    if transposed:
        a = np.swapaxes(a, -1, -2)
    return _numpy_linalg.solve(a, b)


_TRANS_CODES = {0: False, "N": False, 1: True, "T": True, 2: True, "C": True}


def solve_triangular(a, b, trans=0, lower=False, unit_diagonal=False, overwrite_b=False,
                      check_finite=True):
    """The solution `x` of the triangular system `a @ x == b` (or, for `trans` in
    `(1, 2, "T", "C")`, `a.T @ x == b`)."""
    a = np.asarray(a)
    b = np.asarray(b)
    _check_square(a, "a1")
    if trans not in _TRANS_CODES:
        raise ValueError(f"invalid trans argument: {trans!r}")
    if check_finite:
        _check_finite(a, b)
    dtype = _resolve_precision("solve_triangular", a, b)
    a = a.astype(dtype)
    b = b.astype(dtype)
    return _numpy_linalg.solve_triangular(a, b, lower, _TRANS_CODES[trans], unit_diagonal)


def solve_banded(l_and_u, ab, b, overwrite_ab=False, overwrite_b=False, check_finite=True):
    """The solution of the banded system stored in `ab` (LAPACK's compact banded form: row
    `ku - i + j` of column `j` holds `A[i, j]`, for `kl` sub- and `ku` super-diagonals), by
    Gaussian elimination with partial pivoting specialized to the band (Golub & Van Loan §4.3.1).

    The elimination loops over the `n` pivot steps; each step's row swap and elimination touch
    only the entries within the band, as a vectorized slice of the working band matrix.
    """
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
    n = ab.shape[1]
    vector = b.ndim == 1
    x = (b[:, None] if vector else b).astype(dtype).copy()

    # A working band matrix in LAPACK's own expanded storage: `kl` extra scratch rows on top hold
    # the fill-in a pivot swap can carry in from below the original band.
    band = np.zeros((2 * kl + ku + 1, n), dtype=dtype)
    band[kl:, :] = ab

    def row_of(i, j):
        return kl + ku + i - j

    for j in range(n):
        last_row = min(j + kl, n - 1)
        window = np.arange(j, last_row + 1)
        best = window[np.argmax(np.abs(band[row_of(window, j), j]))]
        last_col = min(j + kl + ku, n - 1)
        cols = np.arange(j, last_col + 1)
        if best != j:
            band[row_of(j, cols), cols], band[row_of(best, cols), cols] = (
                band[row_of(best, cols), cols].copy(),
                band[row_of(j, cols), cols].copy(),
            )
            x[[j, best]] = x[[best, j]]
        pivot = band[row_of(j, j), j]
        if pivot == 0.0:
            raise LinAlgError("A singular matrix detected: slice(s) [0] are singular.")
        rows = np.arange(j + 1, last_row + 1)
        elim_cols = np.arange(j + 1, last_col + 1)
        if rows.size:
            factors = band[row_of(rows, j), j] / pivot
            band[row_of(rows, j), j] = factors
            if elim_cols.size:
                row_idx = row_of(rows[:, None], elim_cols[None, :])
                col_idx = np.broadcast_to(elim_cols, row_idx.shape)
                band[row_idx, col_idx] -= factors[:, None] * band[row_of(j, elim_cols), elim_cols][None, :]
            x[rows] -= factors[:, None] * x[j]

    for i in reversed(range(n)):
        last_col = min(i + kl + ku, n - 1)
        cols = np.arange(i + 1, last_col + 1)
        if cols.size:
            x[i] = x[i] - band[row_of(i, cols), cols] @ x[cols]
        x[i] = x[i] / band[row_of(i, i), i]
    return x[:, 0] if vector else x


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
    if assume_a in _POSITIVE_STRUCTURES:
        c = _numpy_linalg.cholesky(a)
        y = _numpy_linalg.solve_triangular(c, _broadcast_identity(a, dtype), True, False, False)
        return _numpy_linalg.solve_triangular(c, y, True, True, False)
    return _numpy_linalg.inv(a)


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
    return _numpy_linalg.det(a)


# -------------------------------------------------------------------------------------------
# lu, lu_factor, lu_solve
# -------------------------------------------------------------------------------------------


def lu_factor(a, overwrite_a=False, check_finite=True):
    """The packed LU factorization LAPACK's `getrf` leaves (`lu`, and 0-based pivot indices
    `piv` such that row `i` was exchanged with row `piv[i]`), or a batch of them."""
    a = np.asarray(a)
    _check_square(a, "a1")
    if check_finite:
        _check_finite(a)
    dtype = _resolve_precision("lu_factor", a)
    a = a.astype(dtype)
    lu, piv = _numpy_linalg.lu(a)
    if np.any(np.diagonal(lu, axis1=-2, axis2=-1) == 0.0):
        warnings.warn("Diagonal number is exactly zero. Singular matrix.", LinAlgWarning, stacklevel=2)
    return lu, piv


def _apply_pivots(piv, b, reverse):
    """Apply (or, `reverse`, undo) the row swaps `piv` encodes to `b` in place: step `k` swaps
    rows `k` and `piv[k]`, so undoing them replays the steps in reverse order."""
    order = range(len(piv) - 1, -1, -1) if reverse else range(len(piv))
    for k in order:
        p = int(piv[k])
        if p != k:
            b[[k, p]] = b[[p, k]]
    return b


def lu_solve(lu_and_piv, b, trans=0, overwrite_b=False, check_finite=True):
    """The solution of `a @ x == b` (or, `trans` in `(1, 2)`, `a.T @ x == b`) given `a`'s packed
    LU factorization from :func:`lu_factor`.

    Composed directly from `lu`'s own storage: the strictly-lower part (unit diagonal) is `L`,
    the upper part (including the diagonal) is `U`, and a row-swap loop applies `piv`.
    """
    lu, piv = lu_and_piv
    lu = np.asarray(lu)
    piv = np.asarray(piv)
    b = np.asarray(b)
    if trans not in (0, 1, 2):
        raise ValueError(f"trans={trans!r} not implemented")
    if check_finite:
        _check_finite(lu, b)
    vector = b.ndim == 1
    x = (b[:, None] if vector else b).astype(lu.dtype).copy()
    if trans == 0:
        x = _apply_pivots(piv, x, reverse=False)
        y = _numpy_linalg.solve_triangular(lu, x, True, False, True)
        x = _numpy_linalg.solve_triangular(lu, y, False, False, False)
    else:
        z = _numpy_linalg.solve_triangular(lu, x, False, True, False)
        w = _numpy_linalg.solve_triangular(lu, z, True, True, True)
        x = _apply_pivots(piv, w, reverse=True)
    return x[:, 0] if vector else x


def _lu_from_packed(packed, piv, permute_l, p_indices):
    m, n = packed.shape
    k = min(m, n)
    l = np.tril(packed[:, :k], -1) + np.eye(m, k, dtype=packed.dtype)
    u = np.triu(packed[:k, :])
    perm = np.arange(m)
    for i, p in enumerate(piv):
        perm[i], perm[p] = perm[p], perm[i]
    indices = np.argsort(perm)
    if permute_l:
        return l[indices], u
    if p_indices:
        return indices.astype(np.int32), l, u
    p_matrix = np.zeros((m, m), dtype=packed.dtype)
    p_matrix[perm, np.arange(m)] = 1.0
    return p_matrix, l, u


def lu(a, permute_l=False, overwrite_a=False, check_finite=True, p_indices=False):
    """The LU factorization of a matrix, or a single stack of them, with explicit pivoting.

    Returns `(p, l, u)` with `a == p @ l @ u` (the default), `(p @ l, u)` if `permute_l`, or
    `(indices, l, u)` if `p_indices` (an int32 row-permutation array with `a == (l @ u)[indices]`).
    """
    a = np.asarray(a)
    if check_finite:
        _check_finite(a)
    dtype = _resolve_precision("lu", a)
    a = a.astype(dtype)
    lu_packed, piv = _numpy_linalg.lu(a)
    if a.ndim == 2:
        return _lu_from_packed(lu_packed, piv, permute_l, p_indices)
    results = [_lu_from_packed(lu_packed[i], piv[i], permute_l, p_indices) for i in range(a.shape[0])]
    return tuple(np.stack(parts) for parts in zip(*results))


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
    l = _numpy_linalg.cholesky(a)
    return l if lower else np.swapaxes(l, -1, -2)


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
    lower = bool(lower)
    y = _numpy_linalg.solve_triangular(c, b, lower, not lower, False)
    return _numpy_linalg.solve_triangular(c, y, lower, lower, False)


# -------------------------------------------------------------------------------------------
# qr
# -------------------------------------------------------------------------------------------

_QR_MODES = {"full": "complete", "qr": "complete", "economic": "reduced"}


def qr(a, overwrite_a=False, lwork=None, mode="full", pivoting=False, check_finite=True):
    """The QR factorization of a matrix, or a stack of them.

    `mode` is one of `"full"`/`"qr"` (the default: `q` is square, `r` is the full `rows x cols`
    shape), `"economic"` (`q` is `rows x k`, `r` is `k x cols`, `k = min(rows, cols)`), `"r"`
    (the full `rows x cols` `r` alone, as a one-element tuple -- unlike `"economic"`, matching
    SciPy's own asymmetry between the two), or `"raw"` (LAPACK's packed reflectors and their
    scale factors, `(h, tau)`, alongside the *reduced* `k x cols` `r`). Column pivoting
    (`pivoting=True`) is not supported.
    """
    if mode not in ("full", "qr", "economic", "r", "raw"):
        raise ValueError("Mode argument should be one of ['full', 'qr', 'r', 'raw', 'economic']")
    if pivoting:
        raise NotImplementedError("scipy.linalg.qr(pivoting=True) is not supported by shellsim's SciPy")
    a = np.asarray(a)
    if check_finite:
        _check_finite(a)
    dtype = _resolve_precision("qr", a)
    a = a.astype(dtype)
    if mode == "raw":
        # `_numpy_linalg`'s own "raw" mode follows NumPy's `linalg.qr(mode="raw")` convention,
        # whose `h` is the *transpose* of LAPACK's (and hence SciPy's own) `geqrf` output shape;
        # swap it back so `h.shape == a.shape`, matching SciPy. Its paired `r` is the reduced
        # `k x cols` triangle, unlike plain `mode="r"`'s full `rows x cols` one.
        h, tau = _numpy_linalg.qr(a, "raw")
        (r,) = _numpy_linalg.qr(a, "r")
        return (np.swapaxes(h, -1, -2), tau), r
    if mode == "r":
        _, r = _numpy_linalg.qr(a, "complete")
        return (r,)
    return _numpy_linalg.qr(a, _QR_MODES[mode])


# -------------------------------------------------------------------------------------------
# eig, eigvals, eigh, eigvalsh
# -------------------------------------------------------------------------------------------


def eig(a, b=None, left=False, right=True, overwrite_a=False, overwrite_b=False,
        check_finite=True, homogeneous_eigvals=False):
    """Eigenvalues and, unless `right` is false, right eigenvectors of a general square matrix,
    or a stack of them. The generalized problem (`b` given) and left eigenvectors are not
    supported."""
    if b is not None:
        raise NotImplementedError(
            "scipy.linalg.eig's generalized problem (b given) is not supported by shellsim's SciPy"
        )
    if left:
        raise NotImplementedError("scipy.linalg.eig(left=True) is not supported by shellsim's SciPy")
    a = np.asarray(a)
    if a.ndim < 2 or a.shape[-1] != a.shape[-2]:
        raise ValueError(f"Expected a square matrix or a batch of square matrices. Got a.shape = {a.shape}")
    if check_finite:
        _check_finite(a)
    dtype = _resolve_precision("eig", a)
    a = a.astype(dtype)
    w, v = _numpy_linalg.eig(a, right)
    return (w, v) if right else w


def eigvals(a, b=None, overwrite_a=False, check_finite=True, homogeneous_eigvals=False):
    """Eigenvalues of a general square matrix, or a stack of them, without eigenvectors. See
    :func:`eig`."""
    return eig(a, b=b, right=False, check_finite=check_finite)


def _eigh_generalized(a, b, lower, compute_vectors):
    """`A x = lambda B x` (`B` symmetric positive definite), reduced to a standard symmetric
    eigenproblem via the Cholesky factor of `B` (Golub & Van Loan §8.7.2, "Problem 1"): with
    `B = L L^T`, solve the standard problem for `C = L^-1 A L^-T`, then map eigenvectors back
    with `v = L^-T y`."""
    a = _symmetrize(a, lower)
    b = _symmetrize(b, lower)
    l = _numpy_linalg.cholesky(b)
    step = _numpy_linalg.solve_triangular(l, a, True, False, False)
    step_t = _numpy_linalg.solve_triangular(l, np.swapaxes(step, -1, -2), True, False, False)
    c = np.swapaxes(step_t, -1, -2)
    w, y = _numpy_linalg.eigh(c, "L", compute_vectors)
    if not compute_vectors:
        return w, None
    return w, _numpy_linalg.solve_triangular(l, y, True, True, False)


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
    dtype = _resolve_precision("eigh", a)
    a = a.astype(dtype)
    if b is not None:
        b = np.asarray(b).astype(dtype)
        if check_finite:
            _check_finite(b)
        w, v = _eigh_generalized(a, b, lower, True)
    else:
        w, v = _numpy_linalg.eigh(a, "L" if lower else "U", True)
    if subset_by_index is None and subset_by_value is None:
        return w if eigvals_only else (w, v)
    # Both kinds of subset select by rank along the last axis, so a stack of matrices keeps the
    # same ranks from every matrix in the stack (`w`'s ranks are already ascending per matrix).
    n = w.shape[-1]
    low, high = 0, n - 1
    if subset_by_index is not None:
        low, high = subset_by_index
        if not (0 <= low <= high <= n - 1):
            raise ValueError(
                "Requested eigenvalue indices are not valid. Valid range is "
                f"[0, {n - 1}] and start <= end, but start={low}, end={high} is given"
            )
    keep = np.zeros(n, dtype=bool)
    keep[low:high + 1] = True
    if subset_by_value is not None:
        value_low, value_high = subset_by_value
        in_range = (w > value_low) & (w < value_high)
        # For a stack, only a rank every matrix in the stack keeps stays -- there is otherwise no
        # single set of ranks that would produce one uniformly shaped result array.
        keep &= in_range if in_range.ndim == 1 else np.all(in_range, axis=tuple(range(in_range.ndim - 1)))
    w = w[..., keep]
    if eigvals_only:
        return w
    return w, v[..., keep]


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
# svd, svdvals
# -------------------------------------------------------------------------------------------


def svd(a, full_matrices=True, compute_uv=True, overwrite_a=False, check_finite=True,
        lapack_driver="gesdd"):
    """The singular value decomposition of a matrix, or a stack of them."""
    if lapack_driver not in ("gesdd", "gesvd"):
        raise ValueError(f'lapack_driver must be "gesdd" or "gesvd", not "{lapack_driver}"')
    a = np.asarray(a)
    if check_finite:
        _check_finite(a)
    dtype = _resolve_precision("svd", a)
    a = a.astype(dtype)
    return _numpy_linalg.svd(a, full_matrices, compute_uv)


def svdvals(a, overwrite_a=False, check_finite=True):
    """The singular values of a matrix, or a stack of them, in descending order."""
    return svd(a, compute_uv=False, check_finite=check_finite)


# -------------------------------------------------------------------------------------------
# lstsq, pinv
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
    driver = lapack_driver or lstsq.default_lapack_driver
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


lstsq.default_lapack_driver = "gelsd"


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


# -------------------------------------------------------------------------------------------
# Subspaces
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


# -------------------------------------------------------------------------------------------
# Matrix exponential
# -------------------------------------------------------------------------------------------

# [13/13] Pade numerator coefficients for e^A, from Higham, "The Scaling and Squaring Method for
# the Matrix Exponential Revisited" (SIAM J. Matrix Anal. Appl., 2005), Table 2.3 / eq. (3.12).
_PADE_13 = [
    64764752532480000.0, 32382376266240000.0, 7771770303897600.0, 1187353796428800.0,
    129060195264000.0, 10559470521600.0, 670442572800.0, 33522128640.0,
    1323241920.0, 40840800.0, 960960.0, 16380.0, 182.0, 1.0,
]
# The scaling threshold for order-13 Pade, theta_13 from Higham (2005), Table 2.3.
_PADE_13_THETA = 5.371920351148152


def expm(a):
    """The matrix exponential of `a`, or a stack of them, by scaling and squaring with a fixed
    [13/13] Pade approximant (see the module-level constants' doc). A fixed Pade order is used at
    every scale rather than Higham's adaptive order selection, which trades a little unneeded
    work for simplicity; accuracy is unaffected because the scaling step already brings the
    matrix norm below the order-13 threshold. For a stack, every matrix is scaled (and later
    squared back) by the same power of two -- the largest any single matrix in the stack needs --
    so the repeated-squaring loop has one shared iteration count; using a larger scale than a
    given matrix strictly needs only improves that matrix's accuracy.
    """
    a = np.asarray(a)
    if a.ndim < 2 or a.shape[-1] != a.shape[-2]:
        raise LinAlgError("Last 2 dimensions of the array must be square")
    dtype = _resolve_precision("expm", a)
    n = a.shape[-1]
    if n == 0:
        return np.zeros(a.shape, dtype=dtype)
    work = a.astype(np.float64)
    norms = np.max(np.sum(np.abs(work), axis=-2), axis=-1)
    over = norms > _PADE_13_THETA
    scaling = int(np.max(np.where(over, np.ceil(np.log2(np.where(over, norms, 1.0) / _PADE_13_THETA)), 0.0)))
    scaled = work / (2.0 ** scaling)

    identity = np.eye(n, dtype=np.float64)
    a2 = scaled @ scaled
    a4 = a2 @ a2
    a6 = a4 @ a2
    c = _PADE_13
    u_inner = c[13] * a6 + c[11] * a4 + c[9] * a2
    u = scaled @ (a6 @ u_inner + c[7] * a6 + c[5] * a4 + c[3] * a2 + c[1] * identity)
    v_inner = c[12] * a6 + c[10] * a4 + c[8] * a2
    v = a6 @ v_inner + c[6] * a6 + c[4] * a4 + c[2] * a2 + c[0] * identity

    result = _numpy_linalg.solve(v - u, v + u)
    for _ in range(scaling):
        result = result @ result
    return result.astype(dtype)


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
    # `result[i, j]` is `c[i - j]` when `i >= j`, else `r[j - i]`: both come from one array of
    # the n+m-1 distinct diagonal values, indexed by the (shifted) diagonal offset `i - j`.
    values = np.concatenate([r[1:][::-1], c])
    indices = np.arange(n)[:, None] - np.arange(m)[None, :] + (m - 1)
    return values[indices].astype(dtype)


def circulant(c):
    """The circulant matrix whose first column is `c`; row `i` is `c` rotated by `i`."""
    c = np.asarray(c).ravel()
    n = len(c)
    indices = (np.arange(n)[:, None] - np.arange(n)[None, :]) % n
    return c[indices]


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


def hilbert(n):
    """The `n x n` Hilbert matrix: entry `(i, j)` is `1 / (i + j + 1)`, 0-based."""
    i = np.arange(1, n + 1, dtype=np.float64)
    return 1.0 / (i[:, None] + i[None, :] - 1.0)
