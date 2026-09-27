"""Cholesky decomposition, following SciPy 1.18's ``scipy/linalg/_decomp_cholesky.py``.

``cholesky`` and ``cho_factor`` call the native batched kernel, which zeroes the unused
triangle in both cases, as SciPy 1.18's does. ``cho_solve`` calls the f2py-style ``potrs``.
The banded forms, ``cholesky_banded`` and ``cho_solve_banded``, are not provided.
"""

import numpy as np
from numpy import asarray, asarray_chkfinite, empty_like
import _scipy_linalg as _batched_linalg
from scipy._lib._util import _apply_over_batch, _asarray_validated, _deprecate_dtypes
from scipy.linalg._misc import LinAlgError, _datacopied, _reject_complex
from scipy.linalg.lapack import (
    _ensure_aligned_and_native,
    _normalize_lapack_dtype,
    get_lapack_funcs,
)

__all__ = ["cholesky", "cho_factor", "cho_solve"]


def _check_format_errors_warnings(routine_name, err_lst):
    msg = (
        f"Internal {routine_name} return info = {[e['lapack_info'] for e in err_lst]} "
        f"for slices {[e['num'] for e in err_lst]}."
    )
    raise LinAlgError(msg)


def _cholesky(a, lower=False, overwrite_a=False, clean=True, check_finite=True):
    """The Cholesky factor of each matrix in ``a``; the first matrix that is not positive
    definite raises ``LinAlgError``."""
    a1 = _asarray_validated(a, check_finite=check_finite)
    a1 = np.atleast_2d(a1)
    if a1.shape[-1] != a1.shape[-2]:
        raise ValueError(f"Expected a square matrix or batch thereof, got {a1.shape=}")
    _deprecate_dtypes("linalg.cholesky", a1)
    a1, overwrite_a = _normalize_lapack_dtype(a1, overwrite_a)
    a1, overwrite_a = _ensure_aligned_and_native(a1, overwrite_a)
    if a1.shape[-1] == 0:
        batch_shape = a1.shape[:-2]
        return np.zeros(batch_shape + (0, 0), dtype=a1.dtype)
    _reject_complex("cholesky", a1)
    c, err_lst = _batched_linalg._cholesky(a1, lower, clean)
    if err_lst:
        _check_format_errors_warnings("potrf", err_lst)
    return c


def cholesky(a, lower=False, overwrite_a=False, check_finite=True):
    """The Cholesky factor of a symmetric positive definite matrix: upper triangular ``U``
    with ``a = U.T @ U``, or lower triangular ``L`` with ``a = L @ L.T`` when ``lower``.

    Only the corresponding triangle of ``a`` is read.
    """
    return _cholesky(
        a, lower=lower, overwrite_a=overwrite_a, clean=True, check_finite=check_finite
    )


def cho_factor(a, lower=False, overwrite_a=False, check_finite=True):
    """``(c, lower)`` for ``cho_solve``: the Cholesky factor and which triangle it is in.

    ``lower`` comes back as a boolean array of the batch shape. Like SciPy 1.18, this reads
    ``a.shape``, so ``a`` must be an array.
    """
    c = _cholesky(
        a, lower=lower, overwrite_a=overwrite_a, clean=False, check_finite=check_finite
    )
    batch_shape = a.shape[:-2]
    ret_lower = np.tile(lower, reps=batch_shape)
    return c, ret_lower


def cho_solve(c_and_lower, b, overwrite_b=False, check_finite=True):
    """Solve ``a @ x = b`` from ``cho_factor(a)``."""
    c, lower = c_and_lower
    return _cho_solve(c, b, lower, overwrite_b=overwrite_b, check_finite=check_finite)


@_apply_over_batch(("c", 2), ("b", "1|2"))
def _cho_solve(c, b, lower, overwrite_b, check_finite):
    if check_finite:
        b1 = asarray_chkfinite(b)
        c = asarray_chkfinite(c)
    else:
        b1 = asarray(b)
        c = asarray(c)
    if c.ndim != 2 or c.shape[0] != c.shape[1]:
        raise ValueError("The factored matrix c is not square.")
    if c.shape[1] != b1.shape[0]:
        raise ValueError(f"incompatible dimensions ({c.shape} and {b1.shape})")
    if b1.size == 0:
        dt = cho_solve((np.eye(2, dtype=b1.dtype), True), np.ones(2, dtype=c.dtype)).dtype
        return empty_like(b1, dtype=dt)
    _reject_complex("cho_solve", c, b1)
    overwrite_b = overwrite_b or _datacopied(b1, b)
    (potrs,) = get_lapack_funcs(("potrs",), (c, b1))
    x, info = potrs(c, b1, lower=lower, overwrite_b=overwrite_b)
    if info != 0:
        raise ValueError(f"illegal value in {-info}th argument of internal potrs")
    return x
