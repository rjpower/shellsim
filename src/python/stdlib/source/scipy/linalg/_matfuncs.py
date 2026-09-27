"""Matrix functions, following SciPy 1.18's ``scipy/linalg/_matfuncs.py``.

``expm`` calls the native port of SciPy's ``_matfuncs_expm.c`` (scaling and squaring with Padé
approximants). ``coshm``, ``sinhm`` and ``tanhm`` are built on it as SciPy builds them.
``khatri_rao`` is SciPy's NumPy code. The functions that need complex arithmetic or a Schur
decomposition (``cosm``, ``sinm``, ``tanm``, ``logm``, ``sqrtm``, ``funm``, ``signm``,
``fractional_matrix_power``) and the Fréchet derivative functions are not provided.
"""

import numpy as np
from _scipy_linalg import matrix_exponential
from scipy._lib._util import _apply_over_batch, _deprecate_dtypes
from scipy.linalg._basic import solve
from scipy.linalg._misc import LinAlgError, _reject_complex

__all__ = ["expm", "coshm", "sinhm", "tanhm", "khatri_rao"]

eps = np.finfo("d").eps
feps = np.finfo("f").eps

_array_precision = {"i": 1, "l": 1, "f": 0, "d": 1, "F": 0, "D": 1}


def _asarray_square(A):
    A = np.asarray(A)
    if len(A.shape) != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("expected square array_like input")
    return A


def _maybe_real(A, B, tol=None):
    """``B``'s real part when ``A`` is real and ``B``'s imaginary part is negligible."""
    if np.isrealobj(A) and np.iscomplexobj(B):
        if tol is None:
            tol = {0: feps * 1e3, 1: eps * 1e6}[_array_precision[B.dtype.char]]
        if np.allclose(B.imag, 0.0, atol=tol):
            B = B.real
    return B


def expm(A):
    """The matrix exponential of a square matrix, or of each matrix in a stack."""
    a = np.asarray(A)
    _deprecate_dtypes("expm", a)
    if a.size == 1 and a.ndim < 2:
        return np.array([[np.exp(a.item())]])
    if a.ndim < 2:
        raise LinAlgError("The input array must be at least two-dimensional")
    if a.shape[-1] != a.shape[-2]:
        raise LinAlgError("Last 2 dimensions of the array must be square")
    if min(*a.shape) == 0:
        dtype = expm(np.eye(2, dtype=a.dtype)).dtype
        return np.empty_like(a, dtype=dtype)
    if a.shape[-2:] == (1, 1):
        return np.exp(a)
    if not np.issubdtype(a.dtype, np.inexact):
        a = a.astype(np.float64)
    elif a.dtype == np.float16:
        a = a.astype(np.float32)
    _reject_complex("expm", a)
    eA, info = matrix_exponential(a)
    if info != 0:
        # An exactly singular Padé denominator, which SciPy reports as a LAPACK error.
        raise RuntimeError(
            "scipy.linalg.expm: Internal LAPACK error during the exponential computation "
            f"(error code {info})"
        )
    return eA


@_apply_over_batch(("A", 2))
def coshm(A):
    """The matrix hyperbolic cosine, ``(expm(A) + expm(-A)) / 2``."""
    A = _asarray_square(A)
    return _maybe_real(A, 0.5 * (expm(A) + expm(-A)))


@_apply_over_batch(("A", 2))
def sinhm(A):
    """The matrix hyperbolic sine, ``(expm(A) - expm(-A)) / 2``."""
    A = _asarray_square(A)
    return _maybe_real(A, 0.5 * (expm(A) - expm(-A)))


@_apply_over_batch(("A", 2))
def tanhm(A):
    """The matrix hyperbolic tangent, ``solve(coshm(A), sinhm(A))``."""
    A = _asarray_square(A)
    return _maybe_real(A, solve(coshm(A), sinhm(A)))


@_apply_over_batch(("a", 2), ("b", 2))
def khatri_rao(a, b):
    """The column-wise Kronecker product of two matrices with the same number of columns."""
    a = np.asarray(a)
    b = np.asarray(b)
    if not (a.ndim == 2 and b.ndim == 2):
        raise ValueError("The both arrays should be 2-dimensional.")
    if not a.shape[1] == b.shape[1]:
        raise ValueError("The number of columns for both arrays should be equal.")
    if a.size == 0 or b.size == 0:
        m = a.shape[0] * b.shape[0]
        n = a.shape[1]
        return np.empty_like(a, shape=(m, n))
    c = a[..., :, np.newaxis, :] * b[..., np.newaxis, :, :]
    return c.reshape((-1,) + c.shape[2:])
