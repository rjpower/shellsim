"""Singular value decomposition and the functions built on it, following SciPy 1.18's
``scipy/linalg/_decomp_svd.py``.

``svd`` and ``svdvals`` compute with ``numpy.linalg.svd``, whichever ``lapack_driver`` is named.
Singular values agree with SciPy's up to rounding, but singular vectors may differ in sign, as
they do between LAPACK drivers, and ``float32`` input is computed in double precision before
rounding.
"""

import numpy as np
from scipy._lib._util import _apply_over_batch, _asarray_validated, _deprecate_dtypes
from scipy.linalg._misc import _reject_complex
from scipy.linalg.lapack import _normalize_lapack_dtype

__all__ = ["svd", "svdvals", "diagsvd", "orth", "subspace_angles", "null_space"]


def svd(
    a, full_matrices=True, compute_uv=True, overwrite_a=False, check_finite=True, lapack_driver="gesdd"
):
    """The singular value decomposition ``a = U @ diag(s) @ Vh`` of a matrix or stack.

    Returns ``(U, s, Vh)``, or ``s`` alone without ``compute_uv``. ``full_matrices=False``
    gives the reduced factors.
    """
    if not isinstance(lapack_driver, str):
        raise TypeError("lapack_driver must be a string")
    if lapack_driver not in ("gesdd", "gesvd"):
        raise ValueError(f'lapack_driver must be "gesdd" or "gesvd", not "{lapack_driver}"')
    a1 = _asarray_validated(a, check_finite=check_finite)
    _deprecate_dtypes("svd", a1)
    if a1.ndim < 2:
        raise ValueError(f"Expected at least ndim=2, got {a1.ndim=}")
    m, n = a1.shape[-2], a1.shape[-1]
    a1, overwrite_a = _normalize_lapack_dtype(a1, overwrite_a)

    if a1.size == 0:
        u0, s0, v0 = svd(np.eye(2, dtype=a1.dtype))
        batch_shape = a1.shape[:-2]
        s = np.empty_like(a1, shape=batch_shape + (0,), dtype=s0.dtype)
        if full_matrices:
            u = np.empty_like(a1, shape=batch_shape + (m, m), dtype=u0.dtype)
            u[...] = np.identity(m)
            v = np.empty_like(a1, shape=batch_shape + (n, n), dtype=v0.dtype)
            v[...] = np.identity(n)
        else:
            u = np.empty_like(a1, shape=batch_shape + (m, 0), dtype=u0.dtype)
            v = np.empty_like(a1, shape=batch_shape + (0, n), dtype=v0.dtype)
        if compute_uv:
            return u, s, v
        return s

    _reject_complex("svd", a1)
    if not compute_uv:
        return np.linalg.svd(a1, compute_uv=False)
    u, s, vh = np.linalg.svd(a1, full_matrices=full_matrices)
    return u, s, vh


def svdvals(a, overwrite_a=False, check_finite=True):
    """The singular values of a matrix, in decreasing order."""
    return svd(a, compute_uv=0, overwrite_a=overwrite_a, check_finite=check_finite)


@_apply_over_batch(("s", 1))
def diagsvd(s, M, N):
    """The ``M x N`` matrix with ``s`` on its diagonal, as ``svd`` returns singular values."""
    part = np.diag(s)
    typ = part.dtype.char
    MorN = len(s)
    if MorN == M:
        return np.hstack((part, np.zeros((M, N - M), dtype=typ)))
    elif MorN == N:
        return np.vstack((part, np.zeros((M - N, N), dtype=typ)))
    raise ValueError("Length of s must be M or N.")


@_apply_over_batch(("A", 2))
def orth(A, rcond=None):
    """An orthonormal basis for the range of ``A``."""
    u, s, vh = svd(A, full_matrices=False)
    M, N = u.shape[0], vh.shape[1]
    if rcond is None:
        rcond = np.finfo(s.dtype).eps * max(M, N)
    tol = np.amax(s, initial=0.0) * rcond
    num = np.sum(s > tol, dtype=int)
    return u[:, :num]


@_apply_over_batch(("A", 2))
def null_space(A, rcond=None, *, overwrite_a=False, check_finite=True, lapack_driver="gesdd"):
    """An orthonormal basis for the null space of ``A``."""
    u, s, vh = svd(
        A,
        full_matrices=True,
        overwrite_a=overwrite_a,
        check_finite=check_finite,
        lapack_driver=lapack_driver,
    )
    M, N = u.shape[0], vh.shape[1]
    if rcond is None:
        rcond = np.finfo(s.dtype).eps * max(M, N)
    tol = np.amax(s, initial=0.0) * rcond
    num = np.sum(s > tol, dtype=int)
    return vh[num:, :].T.conj()


@_apply_over_batch(("A", 2), ("B", 2))
def subspace_angles(A, B):
    """The principal angles between the column spaces of ``A`` and ``B``, largest first."""
    A = _asarray_validated(A, check_finite=True)
    if len(A.shape) != 2:
        raise ValueError(f"expected 2D array, got shape {A.shape}")
    QA = orth(A)
    del A
    B = _asarray_validated(B, check_finite=True)
    if len(B.shape) != 2:
        raise ValueError(f"expected 2D array, got shape {B.shape}")
    if len(B) != len(QA):
        raise ValueError(
            f"A and B must have the same number of rows, got {QA.shape[0]} and {B.shape[0]}"
        )
    QB = orth(B)
    del B
    QA_H_QB = np.dot(QA.T.conj(), QB)
    sigma = svdvals(QA_H_QB)
    if QA.shape[1] >= QB.shape[1]:
        B = QB - np.dot(QA, QA_H_QB)
    else:
        B = QA - np.dot(QB, QA_H_QB.T.conj())
    del QA, QB, QA_H_QB
    mask = sigma**2 >= 0.5
    if mask.any():
        mu_arcsin = np.arcsin(np.clip(svdvals(B, overwrite_a=True), -1.0, 1.0))
    else:
        mu_arcsin = 0.0
    # The smallest sigma belongs to the largest angle.
    return np.where(mask, mu_arcsin, np.arccos(np.clip(sigma[::-1], -1.0, 1.0)))
