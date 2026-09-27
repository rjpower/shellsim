"""The orthogonal Procrustes problem, following SciPy 1.18's ``scipy/linalg/_procrustes.py``."""

import numpy as np
from scipy._lib._util import _apply_over_batch, _asarray_validated
from scipy.linalg._decomp_svd import svd

__all__ = ["orthogonal_procrustes"]


@_apply_over_batch(("A", 2), ("B", 2))
def orthogonal_procrustes(A, B, check_finite=True):
    """``(R, scale)``: the orthogonal ``R`` that best maps ``A`` onto ``B`` (minimizing
    ``||A @ R - B||``), and the sum of the singular values of ``A.T @ B``."""
    A = _asarray_validated(A, check_finite=check_finite)
    B = _asarray_validated(B, check_finite=check_finite)
    if A.ndim != 2:
        raise ValueError(f"expected ndim to be 2, but observed {A.ndim}")
    if A.shape != B.shape:
        raise ValueError(f"the shapes of A and B differ ({A.shape} vs {B.shape})")
    u, w, vt = svd((B.T @ np.conj(A)).T)
    R = u @ vt
    scale = np.sum(w)
    return R, scale
