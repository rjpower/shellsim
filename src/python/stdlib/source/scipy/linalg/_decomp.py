"""Symmetric eigenvalue problems, following SciPy 1.18's ``eigh`` and ``eigvalsh`` in
``scipy/linalg/_decomp.py``.

The standard problem is solved with ``numpy.linalg.eigh``, whichever LAPACK ``driver`` is
named. The generalized problem ``a @ x = w * b @ x`` (and SciPy's types 2 and 3) is reduced to a
standard one with the Cholesky factor of ``b``, as LAPACK's ``sygv`` drivers do. Eigenvalues
agree with SciPy's up to rounding; eigenvectors may differ in sign, as they do between LAPACK
drivers, and ``float32`` input is computed in double precision before rounding. Subsets are
selected from the full spectrum. Only the triangle of ``a`` and ``b`` that ``lower`` names is
read.

``eig``, ``eigvals`` and the banded and tridiagonal eigensolvers are not provided.
"""

import numpy as np
from numpy import inf
from scipy._lib._util import _apply_over_batch, _asarray_validated
from scipy.linalg._misc import LinAlgError, _reject_complex
from scipy.linalg.lapack import get_lapack_funcs

__all__ = ["eigh", "eigvalsh"]


def _symmetric(a, lower):
    """The symmetric matrix whose ``lower`` (or upper) triangle ``a`` holds."""
    if lower:
        return np.tril(a) + np.tril(a, -1).T
    return np.triu(a) + np.triu(a, 1).T


@_apply_over_batch(("a", 2), ("b", 2))
def eigh(
    a,
    b=None,
    *,
    lower=True,
    eigvals_only=False,
    overwrite_a=False,
    overwrite_b=False,
    type=1,
    check_finite=True,
    subset_by_index=None,
    subset_by_value=None,
    driver=None,
):
    """Eigenvalues, ascending, and eigenvectors of a symmetric matrix: ``(w, v)``, or ``w``
    with ``eigvals_only``.

    With ``b``, solves the generalized problem ``a @ v = w * b @ v`` (``type=1``),
    ``a @ b @ v = w * v`` (``type=2``) or ``b @ a @ v = w * v`` (``type=3``) for symmetric
    positive definite ``b``. ``subset_by_index=(lo, hi)`` keeps eigenvalues ``lo`` through
    ``hi``; ``subset_by_value=(lo, hi)`` keeps those in ``(lo, hi]``.
    """
    uplo = "L" if lower else "U"
    drv_str = [None, "ev", "evd", "evr", "evx", "gv", "gvd", "gvx"]
    if driver not in drv_str:
        raise ValueError(
            '"{}" is unknown. Possible values are "None", "{}".'.format(
                driver, '", "'.join(drv_str[1:])
            )
        )
    a1 = _asarray_validated(a, check_finite=check_finite)
    if len(a1.shape) != 2 or a1.shape[0] != a1.shape[1]:
        raise ValueError('expected square "a" matrix')
    if a1.size == 0:
        w_n, v_n = eigh(np.eye(2, dtype=a1.dtype))
        w = np.empty_like(a1, shape=(0,), dtype=w_n.dtype)
        v = np.empty_like(a1, shape=(0, 0), dtype=v_n.dtype)
        if eigvals_only:
            return w
        return w, v
    n = a1.shape[0]

    b1 = None
    if b is not None:
        b1 = _asarray_validated(b, check_finite=check_finite)
        if len(b1.shape) != 2 or b1.shape[0] != b1.shape[1]:
            raise ValueError('expected square "b" matrix')
        if b1.shape != a1.shape:
            raise ValueError(f"wrong b dimensions {b1.shape}, should be {a1.shape}")
        if type not in [1, 2, 3]:
            raise ValueError('"type" keyword only accepts 1, 2, and 3.')

    subset = (subset_by_index is not None) or (subset_by_value is not None)
    if subset_by_index and subset_by_value:
        raise ValueError("Either index or value subset can be requested.")
    if subset_by_index:
        lo, hi = (int(x) for x in subset_by_index)
        if not (0 <= lo <= hi < n):
            raise ValueError(
                "Requested eigenvalue indices are not valid. "
                f"Valid range is [0, {n - 1}] and start <= end, but "
                f"start={lo}, end={hi} is given"
            )
    if subset_by_value:
        lo, hi = subset_by_value
        if not (-inf <= lo < hi <= inf):
            raise ValueError(
                "Requested eigenvalue bounds are not valid. "
                "Valid range is (-inf, inf) and low < high, but "
                f"low={lo}, high={hi} is given"
            )
    if driver:
        if b is None and (driver in ["gv", "gvd", "gvx"]):
            raise ValueError(
                f"{driver} requires input b array to be supplied "
                "for generalized eigenvalue problems."
            )
        if (b is not None) and (driver in ["ev", "evd", "evr", "evx"]):
            raise ValueError(
                f'"{driver}" does not accept input b array for standard eigenvalue problems.'
            )
        if subset and (driver in ["ev", "evd", "gv", "gvd"]):
            raise ValueError(f'"{driver}" cannot compute subsets of eigenvalues')

    _reject_complex("eigh", a1, b1)
    if b1 is None:
        w, v = np.linalg.eigh(a1, UPLO=uplo)
    else:
        w, v = _generalized(a1, b1, lower, type, n)

    if subset_by_index:
        keep = slice(lo, hi + 1)
    elif subset_by_value:
        keep = (w > lo) & (w <= hi)
    else:
        keep = slice(None)
    w = w[keep]
    if eigvals_only:
        return w
    # LAPACK writes the eigenvectors in Fortran order.
    return w, np.asfortranarray(v[:, keep])


def _generalized(a1, b1, lower, itype, n):
    """``(w, v)`` for the generalized problem of type ``itype``, reduced with ``b = L @ L.T``.

    The eigenvectors are normalized as LAPACK normalizes them: ``v.T @ b @ v = I`` for types 1
    and 2 and ``v.T @ inv(b) @ v = I`` for type 3.
    """
    from scipy.linalg._basic import solve_triangular

    (potrf,) = get_lapack_funcs(("potrf",), (b1,))
    c, info = potrf(b1, lower=lower, clean=1)
    if info > 0:
        raise LinAlgError(
            f"The leading minor of order {info} of B is not "
            "positive definite. The factorization of B "
            "could not be completed and no eigenvalues "
            "or eigenvectors were computed."
        )
    factor = c if lower else c.T
    a_sym = _symmetric(a1, lower)
    if itype == 1:
        half = solve_triangular(factor, a_sym, lower=True)
        reduced = solve_triangular(factor, half.T, lower=True)
    else:
        reduced = factor.T @ a_sym @ factor
    w, y = np.linalg.eigh(reduced, UPLO="L")
    if itype == 3:
        v = factor @ y
    else:
        v = solve_triangular(factor, y, lower=True, trans="T")
    return w, v.astype(w.dtype, copy=False)


@_apply_over_batch(("a", 2), ("b", 2))
def eigvalsh(
    a,
    b=None,
    *,
    lower=True,
    overwrite_a=False,
    overwrite_b=False,
    type=1,
    check_finite=True,
    subset_by_index=None,
    subset_by_value=None,
    driver=None,
):
    """The eigenvalues of a symmetric matrix, or of a generalized problem, as ``eigh`` finds
    them."""
    return eigh(
        a,
        b=b,
        lower=lower,
        eigvals_only=True,
        overwrite_a=overwrite_a,
        overwrite_b=overwrite_b,
        type=type,
        check_finite=check_finite,
        subset_by_index=subset_by_index,
        subset_by_value=subset_by_value,
        driver=driver,
    )
