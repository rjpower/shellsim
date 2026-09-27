"""LU decomposition, following SciPy 1.18's ``scipy/linalg/_decomp_lu.py``.

``lu`` calls the native batched kernel; ``lu_factor`` and ``lu_solve`` call the f2py-style
``getrf`` and ``getrs`` wrappers, as SciPy does.
"""

from warnings import warn

import numpy as np
from numpy import asarray, asarray_chkfinite
from _scipy_linalg import _lu as _linalg_lu
from scipy._lib._util import _apply_over_batch, _deprecate_dtypes
from scipy.linalg._misc import LinAlgWarning, _datacopied, _reject_complex
from scipy.linalg.lapack import HAS_ILP64, _normalize_lapack_dtype, get_lapack_funcs

__all__ = ["lu", "lu_solve", "lu_factor"]


@_apply_over_batch(("a", 2))
def lu_factor(a, overwrite_a=False, check_finite=True):
    """The pivoted LU factorization ``(lu, piv)`` of a matrix, for ``lu_solve``.

    ``lu`` holds ``U`` in its upper triangle and the unit lower triangular ``L`` below the
    diagonal; row ``i`` was swapped with row ``piv[i]``. An exactly singular matrix warns with
    ``LinAlgWarning``.
    """
    if check_finite:
        a1 = asarray_chkfinite(a)
    else:
        a1 = asarray(a)
    if a1.size == 0:
        lu = np.empty_like(a1)
        piv = np.arange(0, dtype=np.int64 if HAS_ILP64 else np.int32)
        return lu, piv
    _reject_complex("lu_factor", a1)
    overwrite_a = overwrite_a or (_datacopied(a1, a))
    (getrf,) = get_lapack_funcs(("getrf",), (a1,))
    lu, piv, info = getrf(a1, overwrite_a=overwrite_a)
    if info < 0:
        raise ValueError(f"illegal value in {-info}th argument of internal getrf (lu_factor)")
    if info > 0:
        warn(f"Diagonal number {info} is exactly zero. Singular matrix.", LinAlgWarning, stacklevel=2)
    return lu, piv


def lu_solve(lu_and_piv, b, trans=0, overwrite_b=False, check_finite=True):
    """Solve ``a @ x = b`` from ``lu_factor(a)``; ``trans`` 1 or 2 solves ``a.T @ x = b``."""
    (lu, piv) = lu_and_piv
    return _lu_solve(lu, piv, b, trans=trans, overwrite_b=overwrite_b, check_finite=check_finite)


@_apply_over_batch(("lu", 2), ("piv", 1), ("b", "1|2"))
def _lu_solve(lu, piv, b, trans, overwrite_b, check_finite):
    if check_finite:
        b1 = asarray_chkfinite(b)
    else:
        b1 = asarray(b)
    _deprecate_dtypes("lu_solve", lu, b)
    overwrite_b = overwrite_b or _datacopied(b1, b)
    if lu.shape[0] != b1.shape[0]:
        raise ValueError(f"Shapes of lu {lu.shape} and b {b1.shape} are incompatible")
    if b1.size == 0:
        m = lu_solve((np.eye(2, dtype=lu.dtype), [0, 1]), np.ones(2, dtype=b.dtype))
        return np.empty_like(b1, dtype=m.dtype)
    _reject_complex("lu_solve", lu, b1)
    (getrs,) = get_lapack_funcs(("getrs",), (lu, b1))
    x, info = getrs(lu, piv, b1, trans=trans, overwrite_b=overwrite_b)
    if info == 0:
        return x
    raise ValueError(f"illegal value in {-info}th argument of internal gesv|posv")


def lu(a, permute_l=False, overwrite_a=False, check_finite=True, p_indices=False):
    """The pivoted LU decomposition ``a = P @ L @ U`` of a matrix or stack of matrices.

    Returns ``(P, L, U)``, or ``(P @ L, U)`` with ``permute_l``. With ``p_indices``, ``P`` is
    the row permutation as an index array, so that ``a == (L @ U)[P]``.
    """
    a1 = np.asarray_chkfinite(a) if check_finite else np.asarray(a)
    _deprecate_dtypes("lu", a1)
    if a1.ndim < 2:
        raise ValueError("The input array must be at least two-dimensional.")
    a1, overwrite_a = _normalize_lapack_dtype(a1, overwrite_a)
    *nd, m, n = a1.shape
    k = min(m, n)
    real_dchar = "f" if a1.dtype.char in "fF" else "d"

    if min(*a1.shape) == 0:
        if permute_l:
            PL = np.empty(shape=[*nd, m, k], dtype=a1.dtype)
            U = np.empty(shape=[*nd, k, n], dtype=a1.dtype)
            return PL, U
        P = np.empty([*nd, 0], dtype=np.int32) if p_indices else np.empty([*nd, 0, 0], dtype=real_dchar)
        L = np.empty(shape=[*nd, m, k], dtype=a1.dtype)
        U = np.empty(shape=[*nd, k, n], dtype=a1.dtype)
        return P, L, U

    if a1.shape[-2:] == (1, 1):
        if permute_l:
            return np.ones_like(a1), (a1 if overwrite_a else a1.copy())
        P = np.zeros(shape=[*nd, m], dtype=int) if p_indices else np.ones_like(a1)
        return P, np.ones_like(a1), (a1 if overwrite_a else a1.copy())

    _reject_complex("lu", a1)
    P, L, U = _linalg_lu(a1, permute_l)
    if (not p_indices) and (not permute_l):
        if nd:
            Pa = np.zeros([*nd, m, m], dtype=real_dchar)
            # One-hot encoding of each permutation.
            nd_ix = np.ix_(*([np.arange(x) for x in nd] + [np.arange(m)]))
            Pa[(*nd_ix, P)] = 1
            P = Pa
        else:
            Pa = np.zeros([m, m], dtype=real_dchar)
            Pa[np.arange(m), P] = 1
            P = Pa
    return (L, U) if permute_l else (P, L, U)
