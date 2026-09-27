"""The polar decomposition, as SciPy 1.18's ``scipy/linalg/_decomp_polar.py`` computes it from
the singular value decomposition."""

import numpy as np
from scipy._lib._util import _apply_over_batch
from scipy.linalg._decomp_svd import svd

__all__ = ["polar"]


@_apply_over_batch(("a", 2))
def polar(a, side="right"):
    """``(u, p)`` with ``a = u @ p`` (``side='right'``) or ``a = p @ u`` (``side='left'``),
    where ``u`` has orthonormal columns and ``p`` is symmetric positive semidefinite."""
    if side not in ["right", "left"]:
        raise ValueError("`side` must be either 'right' or 'left'")
    a = np.asarray(a)
    if a.ndim != 2:
        raise ValueError("`a` must be a 2-D array.")
    w, s, vh = svd(a, full_matrices=False)
    u = w.dot(vh)
    if side == "right":
        p = (vh.T.conj() * s).dot(vh)
    else:
        p = (w * s).dot(w.T.conj())
    return u, p
