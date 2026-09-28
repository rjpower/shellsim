"""shellsim's ``numpy.linalg``.

Ten functions (:func:`inv`, :func:`solve`, :func:`det`, :func:`slogdet`, :func:`cholesky`,
:func:`qr`, :func:`eigh`, :func:`eig`, :func:`svd`, plus :func:`eigvals` sharing `eig`'s native
call) are thin wrappers around the native dense-matrix kernels in ``_numpy_linalg``, which batch
over stacked leading dimensions and charge CPU for their cubic (or, for the Jacobi-based
:func:`eigh`/:func:`svd` and the QR-iteration-based :func:`eig`, per-sweep) work before running
it. The remaining functions here (:func:`eigvalsh`, :func:`svdvals`, :func:`matrix_power`,
:func:`matrix_rank`, :func:`pinv`, :func:`lstsq`, :func:`norm`) are plain Python built from those
and from ordinary NumPy array operations. ``_numpy_linalg`` also exposes `lu` and
`solve_triangular`, used only by `scipy.linalg`, not part of this module's own public surface.

See ``docs/numpy.md`` for the algorithm behind each native primitive, its accuracy, and its
(documented, deliberate) differences from real NumPy's LAPACK-backed implementation.
"""

import numpy as np
from _numpy import _LinAlgError as LinAlgError
from _numpy_linalg import cholesky as _cholesky
from _numpy_linalg import det as _det
from _numpy_linalg import eig as _eig
from _numpy_linalg import eigh as _eigh
from _numpy_linalg import inv as _inv
from _numpy_linalg import qr as _qr
from _numpy_linalg import slogdet as _slogdet
from _numpy_linalg import solve as _solve
from _numpy_linalg import svd as _svd

__all__ = [
    "LinAlgError",
    "inv",
    "solve",
    "det",
    "slogdet",
    "cholesky",
    "qr",
    "eigh",
    "eigvalsh",
    "eig",
    "eigvals",
    "svd",
    "svdvals",
    "matrix_power",
    "matrix_rank",
    "pinv",
    "lstsq",
    "norm",
]


def inv(a):
    """The multiplicative inverse of a square matrix, or a stack of them."""
    return _inv(a)


def solve(a, b):
    """The solution ``x`` of ``a @ x == b``."""
    return _solve(a, b)


def det(a):
    """The determinant of a square matrix, or a stack of them."""
    return _det(a)


def slogdet(a):
    """The sign and the natural log of the absolute value of the determinant."""
    return _slogdet(a)


def cholesky(a):
    """The lower-triangular Cholesky factor ``L`` with ``L @ L.T == a``."""
    return _cholesky(a)


def qr(a, mode="reduced"):
    """The QR factorization of a matrix, or a stack of them.

    ``mode`` follows real NumPy: ``"reduced"`` (default), ``"complete"``, ``"r"`` (``R`` only),
    or ``"raw"`` (the packed Householder reflectors and their scale factors, as LAPACK's
    ``geqrf`` leaves them).
    """
    return _qr(a, mode)


def eigh(a, UPLO="L"):
    """Eigenvalues (ascending) and orthonormal eigenvectors of a Hermitian matrix.

    Only the triangle named by ``UPLO`` ("L" for lower, the default, or "U" for upper) is read;
    the other is assumed to mirror it. Eigenvector signs are normalized so each column's
    largest-magnitude entry is positive (ties keep the first such entry positive); real NumPy's
    signs come from LAPACK and are not otherwise comparable.
    """
    return _eigh(a, UPLO, True)


def eigvalsh(a, UPLO="L"):
    """Eigenvalues (ascending) of a Hermitian matrix, without eigenvectors."""
    w, _ = _eigh(a, UPLO, False)
    return w


def eig(a):
    """Eigenvalues and right eigenvectors of a general square matrix, or a stack of them.

    Returns ``(w, v)``: `w` holds the eigenvalues (not necessarily ordered, and not necessarily
    real even when `a` is real: a real matrix's complex eigenvalues occur in conjugate pairs),
    and column ``v[:, i]`` is the eigenvector for ``w[i]``, normalized to unit length. `w` and `v`
    are always complex, matching real NumPy's own `eig` (whose docs describe casting an
    all-real result down to a real dtype, but whose current implementation, like this one, does
    not). Eigenvector signs (more generally, for a complex eigenvector, its phase) are normalized
    so each column's largest-magnitude entry is a positive real number; real NumPy's come from
    LAPACK and are not otherwise comparable.

    Raises :class:`LinAlgError` if the underlying QR iteration fails to converge, or if `a` is
    not square.
    """
    return _eig(a, True)


def eigvals(a):
    """Eigenvalues of a general square matrix, or a stack of them, without eigenvectors. See
    :func:`eig`."""
    w, _ = _eig(a, False)
    return w


def svd(a, full_matrices=True, compute_uv=True):
    """The singular value decomposition of a matrix, or a stack of them.

    Singular values are descending. When `compute_uv` is true, returns ``(u, s, vh)`` with
    ``u @ diag(s) @ vh`` reconstructing `a` (restricted to the leading `k = min(rows, cols)`
    columns of `u` and rows of `vh` when ``full_matrices`` is false). Singular-vector signs are
    normalized the same way :func:`eigh`'s eigenvector signs are.
    """
    return _svd(a, full_matrices, compute_uv)


def svdvals(a):
    """The singular values of a matrix, or a stack of them, in descending order."""
    return _svd(a, True, False)


def matrix_power(a, n):
    """``a`` raised to the integer power `n` by repeated squaring; `n < 0` uses :func:`inv`."""
    a = np.asarray(a)
    n = int(n)
    if n == 0:
        return np.broadcast_to(np.eye(a.shape[-1], dtype=a.dtype), a.shape).copy()
    if n < 0:
        a = inv(a)
        n = -n
    result = None
    base = a
    while n > 0:
        if n & 1:
            result = base if result is None else result @ base
        n >>= 1
        if n:
            base = base @ base
    return result


def matrix_rank(a, tol=None, hermitian=False):
    """The number of singular values of `a` above a shape- and magnitude-derived tolerance.

    If `hermitian`, the (real) eigenvalues' magnitudes are used instead of the singular values,
    which is cheaper and exact for a genuinely Hermitian `a`.
    """
    a = np.asarray(a)
    if a.ndim < 2:
        return int(np.any(a))
    if hermitian:
        s = np.sort(np.abs(eigvalsh(a)), axis=-1)[..., ::-1]
    else:
        s = svdvals(a)
    if tol is None:
        largest = np.amax(s, axis=-1, keepdims=True)
        tol = largest * max(a.shape[-2:]) * np.finfo(s.dtype).eps
    return np.sum(s > tol, axis=-1)


def pinv(a, rcond=1e-15, hermitian=False):
    """The Moore-Penrose pseudo-inverse of `a`, computed from its SVD."""
    a = np.asarray(a)
    if a.dtype != np.float32:
        a = a.astype(np.float64)
    u, s, vt = svd(a, full_matrices=False)
    cutoff = rcond * np.amax(s, axis=-1, keepdims=True)
    large = s > cutoff
    safe = np.where(large, s, 1.0)
    s_inv = np.where(large, 1.0 / safe, 0.0)
    v = np.swapaxes(vt, -1, -2)
    ut = np.swapaxes(u, -1, -2)
    return (v * s_inv[..., np.newaxis, :]) @ ut


def lstsq(a, b, rcond=None):
    """The least-squares solution of ``a @ x == b``, via `a`'s SVD.

    Returns ``(x, residuals, rank, singular_values)``. `residuals` holds the sum of squared
    residuals per right-hand-side column when `a` has more rows than columns and full column
    rank; otherwise it is an empty array, as real NumPy's does.
    """
    a = np.asarray(a)
    if a.dtype != np.float32:
        a = a.astype(np.float64)
    b = np.asarray(b)
    m, n = a.shape
    is_1d = b.ndim == 1
    b2 = b[:, np.newaxis] if is_1d else b
    u, s, vt = svd(a, full_matrices=False)
    if rcond is None:
        rcond = np.finfo(s.dtype).eps * max(m, n)
    cutoff = rcond * s[0] if s.size else 0.0
    large = s > cutoff
    rank = int(np.sum(large))
    safe = np.where(large, s, 1.0)
    s_inv = np.where(large, 1.0 / safe, 0.0)
    utb = np.swapaxes(u, -1, -2) @ b2
    x = np.swapaxes(vt, -1, -2) @ (s_inv[:, np.newaxis] * utb)
    if m > n and rank == n:
        residuals = np.sum((b2 - a @ x) ** 2, axis=0)
    else:
        residuals = np.empty(0)
    if is_1d:
        x = x[:, 0]
    return x, residuals, rank, s


def _vector_norm(x, ord, axis, keepdims):
    if ord is None or ord == 2:
        return np.sqrt(np.sum(x * x, axis=axis, keepdims=keepdims))
    if ord == np.inf:
        return np.max(np.abs(x), axis=axis, keepdims=keepdims)
    if ord == -np.inf:
        return np.min(np.abs(x), axis=axis, keepdims=keepdims)
    if ord == 0:
        return np.sum(x != 0, axis=axis, keepdims=keepdims).astype(np.float64)
    if ord == 1:
        return np.sum(np.abs(x), axis=axis, keepdims=keepdims)
    return np.sum(np.abs(x) ** ord, axis=axis, keepdims=keepdims) ** (1.0 / ord)


def _matrix_norm(x, ord, keepdims):
    if ord is None or ord == "fro":
        result = np.sqrt(np.sum(x * x, axis=(-2, -1)))
    elif ord == "nuc":
        result = np.sum(svdvals(x), axis=-1)
    elif ord == 1:
        result = np.max(np.sum(np.abs(x), axis=-2), axis=-1)
    elif ord == -1:
        result = np.min(np.sum(np.abs(x), axis=-2), axis=-1)
    elif ord == np.inf:
        result = np.max(np.sum(np.abs(x), axis=-1), axis=-1)
    elif ord == -np.inf:
        result = np.min(np.sum(np.abs(x), axis=-1), axis=-1)
    elif ord == 2:
        result = np.amax(svdvals(x), axis=-1)
    elif ord == -2:
        result = np.amin(svdvals(x), axis=-1)
    else:
        raise ValueError("Invalid norm order for matrices.")
    if keepdims:
        result = result[..., np.newaxis, np.newaxis]
    return result


def norm(x, ord=None, axis=None, keepdims=False):
    """A vector or matrix norm of `x`.

    With `axis` omitted, a 1-D `x` gets a vector norm and a 2-D `x` gets a matrix norm; with a
    single integer `axis`, a vector norm is taken along that axis (batching over the rest); with
    a 2-tuple `axis`, a matrix norm is taken over those two axes (batching over the rest). `ord`
    selects the kind of norm: for vectors, ``None``/``2`` (Euclidean), ``1``, ``0`` (count of
    nonzeros), `inf`/``-inf``, or any other real `p` (an Lp norm); for matrices, ``None``/``"fro"``
    (Frobenius), ``"nuc"`` (nuclear, the singular values' sum), ``1``/``-1``/`inf`/``-inf`` (the
    largest or smallest absolute column or row sum), or ``2``/``-2`` (the largest or smallest
    singular value).
    """
    x = np.asarray(x)
    real_dtype = np.float32 if x.dtype == np.float32 else np.float64
    xf = x.astype(np.float64)
    if axis is None:
        if xf.ndim == 1:
            result = _vector_norm(xf, ord, None, keepdims)
        elif xf.ndim == 2:
            result = _matrix_norm(xf, ord, keepdims)
        else:
            raise ValueError(
                "Improper number of dimensions to norm."
            )
    elif isinstance(axis, tuple):
        if len(axis) != 2:
            raise ValueError("Invalid norm order for matrices.")
        row_axis, col_axis = (a % xf.ndim for a in axis)
        moved = np.moveaxis(xf, (row_axis, col_axis), (-2, -1))
        result = _matrix_norm(moved, ord, keepdims)
    else:
        result = _vector_norm(xf, ord, axis, keepdims)
    return result.astype(real_dtype)
