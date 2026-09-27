"""QR decomposition, following SciPy 1.18's ``scipy/linalg/_decomp_qr.py``.

``qr`` calls the native batched kernel, which ports LAPACK's Householder QR (``geqrf`` and
``orgqr``) and its column-pivoted form (``geqp3``). ``qr_multiply`` and ``rq`` are not provided.
"""

import warnings

import numpy as np
import _scipy_linalg as _batched_linalg
from scipy._lib._util import _deprecate_dtypes
from scipy.linalg._basic import _format_emit_errors_warnings
from scipy.linalg._misc import _reject_complex
from scipy.linalg.lapack import HAS_ILP64, _normalize_lapack_dtype

__all__ = ["qr"]

# SciPy's `_NoValue` sentinel, which tells an omitted `lwork` from an explicit `None`.
_NoValue = object()


def qr(a, overwrite_a=False, lwork=_NoValue, mode="full", pivoting=False, check_finite=True):
    """The QR decomposition ``a = Q @ R`` of a matrix or stack of matrices.

    ``mode`` is ``'full'`` (``Q`` is ``M x M``), ``'economic'`` (``Q`` is ``M x K`` and ``R``
    is ``K x N``, for ``K = min(M, N)``), ``'r'`` (``R`` alone) or ``'raw'`` (LAPACK's
    Householder factors and ``tau`` in place of ``Q``). With ``pivoting``, the column
    permutation ``P`` (0-based ``int32``) comes last, and ``a[:, P] = Q @ R``.
    """
    modes = {"full": 1, "qr": 1, "r": 11, "raw": 21, "economic": 31}
    if mode not in modes.keys():
        raise ValueError(f"Mode argument should be one of {list(modes.keys())}")
    modeFlag = modes[mode]

    if check_finite:
        a1 = np.asarray_chkfinite(a)
    else:
        a1 = np.asarray(a)
    _deprecate_dtypes("linalg.qr", a1)
    if a1.ndim < 2:
        raise ValueError("Expected at least a 2-D array")
    M, N = a1.shape[-2], a1.shape[-1]

    if lwork is not _NoValue:
        if lwork is not None and lwork != -1 and lwork <= M:
            raise ValueError(f"lwork should be None, -1 or > M, got {lwork}")
        warnings.warn(
            "scipy.linalg: the `lwork` keyword is deprecated and no longer in use"
            " as of SciPy 1.18.0 and will be removed in SciPy 1.20.0",
            DeprecationWarning,
            stacklevel=2,
        )

    a1, overwrite_a = _normalize_lapack_dtype(a1, overwrite_a)
    if a1.size == 0:
        K = min(M, N)
        batch_shape = a1.shape[:-2]
        if mode not in ["economic", "raw"]:
            Q = np.empty_like(a1, shape=batch_shape + (M, M))
            Q[..., :, :] = np.identity(M)
            R = np.empty_like(a1)
        else:
            Q = np.empty_like(a1, shape=batch_shape + (M, K))
            R = np.empty_like(a1, shape=batch_shape + (K, N))
        if pivoting:
            Rj = R, np.arange(N, dtype=np.int64 if HAS_ILP64 else np.int32)
        else:
            Rj = (R,)
        if mode == "r":
            return Rj
        elif mode == "raw":
            qr = np.empty_like(a1, shape=batch_shape + (M, N))
            tau = np.zeros_like(a1, shape=batch_shape + (K,))
            return ((qr, tau),) + Rj
        return (Q,) + Rj

    _reject_complex("qr", a1)
    Q, R, tau, jpvt, err_lst = _batched_linalg._qr(a1, modeFlag, pivoting)
    if err_lst:
        _format_emit_errors_warnings(err_lst)
    if pivoting:
        Rj = R, jpvt
    else:
        Rj = (R,)
    if modeFlag == modes["raw"]:
        # geqrf's factor, in the Fortran order f2py returns for a single matrix.
        Q = (np.asfortranarray(Q) if Q.ndim == 2 else Q, tau)
    elif modeFlag == modes["economic"] and M < N:
        Q = Q[..., :, :M]
    if modeFlag == modes["r"]:
        return Rj
    return (Q,) + Rj
