"""``LinAlgError``, ``LinAlgWarning``, ``norm`` and ``bandwidth``, following SciPy 1.18's
``scipy/linalg/_misc.py``, and ``issymmetric`` and ``ishermitian``, which SciPy writes in Cython
in ``_cythonized_array_utils.pyx``."""

import numpy as np
from numpy.linalg import LinAlgError
from _scipy_linalg import _bandwidth
from scipy._lib._util import _apply_over_batch, _deprecate_dtypes
from scipy.linalg.blas import get_blas_funcs
from scipy.linalg.lapack import get_lapack_funcs

__all__ = ["LinAlgError", "LinAlgWarning", "norm", "bandwidth", "issymmetric", "ishermitian"]


class LinAlgWarning(RuntimeWarning):
    """The warning SciPy's linear algebra functions emit for ill-conditioned input."""


def norm(a, ord=None, axis=None, keepdims=False, check_finite=True):
    """A matrix or vector norm, as ``numpy.linalg.norm`` computes it after a finiteness check.

    Like SciPy, the Euclidean norm of a vector uses BLAS ``nrm2``, and the 1- and infinity
    norms of a Fortran-ordered matrix use LAPACK ``lange``.
    """
    if check_finite:
        a = np.asarray_chkfinite(a)
    else:
        a = np.asarray(a)
    _deprecate_dtypes("norm", a)
    if a.size and a.dtype.char in "fdFD" and axis is None and not keepdims:
        if ord in (None, 2) and (a.ndim == 1):
            nrm2 = get_blas_funcs("nrm2", dtype=a.dtype, ilp64="preferred")
            return nrm2(a)
        if a.ndim == 2:
            lange_args = None
            if ord == 1:
                if np.isfortran(a):
                    lange_args = "1", a
                elif np.isfortran(a.T):
                    lange_args = "i", a.T
            elif ord == np.inf:
                if np.isfortran(a):
                    lange_args = "i", a
                elif np.isfortran(a.T):
                    lange_args = "1", a.T
            if lange_args:
                lange = get_lapack_funcs("lange", dtype=a.dtype, ilp64="preferred")
                return lange(*lange_args)
    return np.linalg.norm(a, ord=ord, axis=axis, keepdims=keepdims)


def _datacopied(arr, original):
    """Whether ``arr`` is a fresh copy of ``original`` that may be overwritten."""
    if arr is original:
        return False
    if not isinstance(original, np.ndarray) and hasattr(original, "__array__"):
        return False
    return arr.base is None


def bandwidth(a):
    """The lower and upper bandwidths of a matrix, or of each matrix in a stack.

    The lower bandwidth is the largest ``i - j`` of a nonzero ``a[i, j]``, and the upper
    bandwidth the largest ``j - i``. They are Python ints for a 2-d array and ``int64`` arrays
    of the batch shape otherwise.

    >>> bandwidth(np.array([[1, 2, 0], [0, 1, 2], [3, 0, 1]]))
    (2, 1)
    """
    a = np.asarray(a)
    if a.ndim < 2:
        raise ValueError("Input array must be at least 2D.")
    if a.dtype == np.float16:
        raise TypeError(f"Input array with {a.dtype} dtype is not supported.")
    elif not np.isdtype(a.dtype, ("numeric", "bool")):
        raise TypeError(f"Input array must have a numeric dtype, got {a.dtype}.")
    if a.size == 0:
        if a.ndim == 2:
            return (np.int64(0), np.int64(0))
        batch_shape = a.shape[:-2]
        return (np.zeros(batch_shape, dtype=np.int64), np.zeros(batch_shape, dtype=np.int64))
    return _bandwidth(a != 0)


def _check_square(a):
    if a.ndim != 2:
        raise ValueError("Input array must be a 2D NumPy array.")
    if not np.equal(*a.shape):
        raise ValueError("Input array must be square.")


def _exact_comparison(a):
    """Whether ``issymmetric`` and ``ishermitian`` compare ``a`` exactly rather than with
    ``allclose``, rejecting the dtypes SciPy's Cython kernels have no signature for."""
    if a.dtype.char not in np.typecodes["AllInteger"] + "?fdFD":
        raise TypeError("No matching signature found")


@_apply_over_batch(("a", 2))
def issymmetric(a, atol=None, rtol=None):
    """Whether a square matrix equals its transpose.

    Without tolerances the off-diagonal elements are compared exactly, so a NaN off the
    diagonal is never symmetric and the diagonal is not examined. With ``atol`` or ``rtol``
    the comparison is ``numpy.allclose``.
    """
    _check_square(a)
    if a.size == 0:
        return True
    if (atol or rtol) and not np.issubdtype(a.dtype, np.integer):
        return np.allclose(a, a.T, atol=atol if atol else 0.0, rtol=rtol if rtol else 0.0)
    _exact_comparison(a)
    diagonal = np.eye(a.shape[0], dtype=bool)
    return bool(np.all((a == a.T) | diagonal))


@_apply_over_batch(("a", 2))
def ishermitian(a, atol=None, rtol=None):
    """Whether a square matrix equals its conjugate transpose.

    For complex input the diagonal is compared too, so it must be real. Tolerances work as in
    ``issymmetric``.
    """
    _check_square(a)
    if a.size == 0:
        return True
    if (atol or rtol) and not np.issubdtype(a.dtype, np.integer):
        return np.allclose(a, a.conj().T, atol=atol if atol else 0.0, rtol=rtol if rtol else 0.0)
    _exact_comparison(a)
    if np.iscomplexobj(a):
        return bool(np.all(a == a.conj().T))
    diagonal = np.eye(a.shape[0], dtype=bool)
    return bool(np.all((a == a.T) | diagonal))


def _reject_complex(name, *arrays):
    """Raise ``NotImplementedError`` for complex ``arrays``, which shellsim's kernels lack."""
    if any(a is not None and np.iscomplexobj(a) for a in arrays):
        raise NotImplementedError(
            f"complex input to scipy.linalg.{name} is not supported by shellsim's SciPy"
        )
