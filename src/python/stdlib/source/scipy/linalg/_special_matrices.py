"""Special matrix constructors, following SciPy 1.18's ``scipy/linalg/_special_matrices.py``.

SciPy builds Toeplitz, circulant and Hankel matrices as strided views of one vector; shellsim
gathers the same elements with index arrays, since it has no ``as_strided``. The other
constructors are SciPy's code, with ``array_api_extra`` helpers replaced by their NumPy forms.
"""

import math
import warnings

import numpy as np
from scipy._lib._util import _apply_over_batch

__all__ = [
    "toeplitz",
    "circulant",
    "hankel",
    "hadamard",
    "leslie",
    "block_diag",
    "companion",
    "helmert",
    "hilbert",
    "invhilbert",
    "pascal",
    "invpascal",
    "dft",
    "fiedler",
    "fiedler_companion",
    "convolution_matrix",
]


def toeplitz(c, r=None):
    """The Toeplitz matrix with first column ``c`` and first row ``r`` (whose first element
    is ignored); ``r`` defaults to ``conj(c)``."""
    c = np.atleast_1d(c)
    if r is None:
        r = c.conjugate()
    else:
        r = np.atleast_1d(r)
    return _toeplitz(c, r)


@_apply_over_batch(("c", 1), ("r", 1))
def _toeplitz(c, r):
    # `vals` is reversed `c` followed by `r[1:]`; element (i, j) is `vals[len(c) - 1 - i + j]`.
    vals = np.concatenate((c[::-1], r[1:]))
    index = (len(c) - 1) - np.arange(len(c))[:, np.newaxis] + np.arange(len(r))
    return vals[index]


def circulant(c):
    """The circulant matrix with first column ``c``: element (i, j) is ``c[(i - j) % n]``.
    Leading axes of ``c`` are batch axes."""
    c = np.atleast_1d(c)
    N = c.shape[-1]
    index = (np.arange(N)[:, np.newaxis] - np.arange(N)) % N
    return c[..., index]


def hankel(c, r=None):
    """The Hankel matrix with first column ``c`` and last row ``r`` (whose first element is
    ignored); ``r`` defaults to zeros."""
    c = np.asarray(c)
    if r is None:
        r = np.zeros_like(c)
    else:
        r = np.asarray(r)
    if c.ndim > 1 or r.ndim > 1:
        msg = (
            "Beginning in SciPy 1.19, multidimensional input will be treated as a "
            "batch, not `ravel`ed. To preserve the existing behavior and silence "
            "this warning, `ravel` arguments before passing them to `hankel`."
        )
        warnings.warn(msg, FutureWarning, stacklevel=2)
        c, r = c.ravel(), r.ravel()
    # Element (i, j) is `vals[i + j]`.
    vals = np.concatenate((c, r[1:]))
    index = np.arange(len(c))[:, np.newaxis] + np.arange(len(r))
    return vals[index]


def hadamard(n, dtype=int):
    """The ``n x n`` Hadamard matrix of Sylvester's construction, for ``n`` a power of 2."""
    if n < 1:
        lg2 = 0
    else:
        lg2 = int(math.log(n, 2))
    if 2**lg2 != n:
        raise ValueError("n must be a positive integer, and n must be a power of 2")
    H = np.array([[1]], dtype=dtype)
    for i in range(0, lg2):
        H = np.vstack((np.hstack((H, H)), np.hstack((H, -H))))
    return H


@_apply_over_batch(("f", 1), ("s", 1))
def leslie(f, s):
    """The Leslie matrix with fecundity coefficients ``f`` and survival coefficients ``s``."""
    f = np.atleast_1d(f)
    s = np.atleast_1d(s)
    if f.shape[-1] != s.shape[-1] + 1:
        raise ValueError(
            "Incorrect lengths for f and s. The length of s along "
            "the last axis must be one less than the length of f."
        )
    if s.shape[-1] == 0:
        raise ValueError("The length of s must be at least 1.")
    n = f.shape[-1]
    tmp = f[0] + s[0]
    a = np.zeros((n, n), dtype=tmp.dtype)
    a[0] = f
    a[list(range(1, n)), list(range(0, n - 1))] = s
    return a


def block_diag(*arrs):
    """The block diagonal matrix of ``arrs``; lower-dimensional inputs count as ``1 x n``."""
    if arrs == ():
        arrs = ([],)
    arrs = [np.atleast_2d(np.asarray(a)) for a in arrs]
    batch_shapes = [a.shape[:-2] for a in arrs]
    batch_shape = np.broadcast_shapes(*batch_shapes)
    arrs = [np.broadcast_to(a, batch_shape + a.shape[-2:]) for a in arrs]
    out_dtype = np.result_type(*arrs)
    block_shapes = [a.shape[-2:] for a in arrs]
    out = np.zeros(
        batch_shape + tuple(map(int, np.sum(np.asarray(block_shapes), axis=0))), dtype=out_dtype
    )
    r, c = 0, 0
    for i, (rr, cc) in enumerate(block_shapes):
        out[..., r : r + rr, c : c + cc] = arrs[i]
        r += rr
        c += cc
    return out


def companion(a):
    """The companion matrix of the polynomial with coefficients ``a``, highest degree first."""
    a = np.atleast_1d(a)
    n = a.shape[-1]
    if n < 2:
        raise ValueError("The length of `a` along the last axis must be at least 2.")
    if np.any(a[..., 0] == 0):
        raise ValueError(
            "The first coefficient(s) of `a` (i.e. elements "
            "of `a[..., 0]`) must not be zero."
        )
    first_row = -a[..., 1:] / (1.0 * a[..., 0:1])
    c = np.zeros(a.shape[:-1] + (n - 1, n - 1), dtype=first_row.dtype)
    c[..., 0, :] = first_row
    c[..., np.arange(1, n - 1), np.arange(0, n - 2)] = 1
    return c


def helmert(n, full=False):
    """The Helmert matrix of order ``n``, without its first row unless ``full``."""
    H = np.tril(np.ones((n, n)), -1) - np.diag(np.arange(n))
    d = np.arange(n) * np.arange(1, n + 1)
    H[0] = 1
    d[0] = n
    H_full = H / np.sqrt(d)[:, np.newaxis]
    if full:
        return H_full
    return H_full[1:]


def hilbert(n):
    """The Hilbert matrix of order ``n``: element (i, j) is ``1 / (i + j + 1)``."""
    values = 1.0 / (1.0 + np.arange(2 * n - 1))
    return hankel(values[:n], r=values[n - 1 :])


def invhilbert(n, exact=False):
    """The inverse of the Hilbert matrix of order ``n``; with ``exact``, as integers (Python
    ints in an object array for ``n > 14``)."""
    from scipy.special import comb

    if exact:
        if n > 14:
            dtype = object
        else:
            dtype = np.int64
    else:
        dtype = np.float64
    invh = np.empty((n, n), dtype=dtype)
    for i in range(n):
        for j in range(0, i + 1):
            s = i + j
            invh[i, j] = (
                (-1) ** s
                * (s + 1)
                * comb(n + i, n - j - 1, exact=exact)
                * comb(n + j, n - i - 1, exact=exact)
                * comb(s, i, exact=exact) ** 2
            )
            if i != j:
                invh[j, i] = invh[i, j]
    return invh


def pascal(n, kind="symmetric", exact=True):
    """The ``n x n`` Pascal matrix: ``'symmetric'``, ``'lower'`` or ``'upper'``."""
    from scipy.special import comb

    if kind not in ["symmetric", "lower", "upper"]:
        raise ValueError("kind must be 'symmetric', 'lower', or 'upper'")
    if exact:
        if n >= 35:
            L_n = np.empty((n, n), dtype=object)
            L_n.fill(0)
        else:
            L_n = np.zeros((n, n), dtype=np.uint64)
        for i in range(n):
            for j in range(i + 1):
                L_n[i, j] = comb(i, j, exact=True)
    else:
        L_n = comb(np.arange(n)[:, np.newaxis], np.arange(n)[np.newaxis, :])
    if kind == "lower":
        p = L_n
    elif kind == "upper":
        p = L_n.T
    else:
        p = np.dot(L_n, L_n.T)
    return p


def invpascal(n, kind="symmetric", exact=True):
    """The inverse of the ``n x n`` Pascal matrix of ``kind``."""
    from scipy.special import comb

    if kind not in ["symmetric", "lower", "upper"]:
        raise ValueError("'kind' must be 'symmetric', 'lower' or 'upper'.")
    if kind == "symmetric":
        if exact:
            if n > 34:
                dt = object
            else:
                dt = np.int64
        else:
            dt = np.float64
        invp = np.empty((n, n), dtype=dt)
        for i in range(n):
            for j in range(0, i + 1):
                v = 0
                for k in range(n - i):
                    v += comb(i + k, k, exact=exact) * comb(i + k, i + k - j, exact=exact)
                invp[i, j] = (-1) ** (i - j) * v
                if i != j:
                    invp[j, i] = invp[i, j]
    else:
        # Inverting a triangular Pascal matrix changes the sign of every other diagonal.
        invp = pascal(n, kind=kind, exact=exact)
        if invp.dtype == np.uint64:
            # The values are far below 2**63, so the reinterpretation is exact.
            invp = invp.view(np.int64)
        invp *= toeplitz((-1) ** np.arange(n)).astype(invp.dtype)
    return invp


def dft(n, scale=None):
    """The ``n x n`` discrete Fourier transform matrix, scaled by ``1/sqrt(n)`` (``'sqrtn'``)
    or ``1/n`` (``'n'``) if asked."""
    if scale not in [None, "sqrtn", "n"]:
        raise ValueError(f"scale must be None, 'sqrtn', or 'n'; {scale!r} is not valid.")
    omegas = np.exp(-2j * np.pi * np.arange(n) / n).reshape(-1, 1)
    m = omegas ** np.arange(n)
    if scale == "sqrtn":
        m /= math.sqrt(n)
    elif scale == "n":
        m /= n
    return m


def fiedler(a):
    """The Fiedler matrix of ``a``: element (i, j) is ``|a[i] - a[j]|``."""
    a = np.atleast_1d(np.asarray(a))
    if a.size == 0:
        return np.empty((0, 0), dtype=np.float64)
    elif a.size == 1:
        return np.asarray([[0.0]])
    return np.abs(a[..., :, np.newaxis] - a[..., np.newaxis, :])


def fiedler_companion(a):
    """Fiedler's pentadiagonal companion matrix of the polynomial with coefficients ``a``."""
    a = np.atleast_1d(a)
    if a.ndim > 1:
        return np.apply_along_axis(fiedler_companion, -1, a)
    if a.size <= 2:
        if a.size == 2:
            return np.array([[-(a / a[0])[-1]]])
        if a.size == 1:
            return np.empty((0, 0), dtype=a.dtype)
        return np.array([], dtype=a.dtype)
    if a[0] == 0.0:
        raise ValueError("Leading coefficient is zero.")
    a = a / a[0]
    n = a.size - 1
    c = np.zeros((n, n), dtype=a.dtype)
    # subdiagonals
    c[range(3, n, 2), range(1, n - 2, 2)] = 1.0
    c[range(2, n, 2), range(1, n - 1, 2)] = -a[3::2]
    # superdiagonals
    c[range(0, n - 2, 2), range(2, n, 2)] = 1.0
    c[range(0, n - 1, 2), range(1, n, 2)] = -a[2::2]
    c[[0, 1], 0] = [-a[1], 1]
    return c


def convolution_matrix(a, n, mode="full"):
    """The matrix whose product with a vector of length ``n`` is ``np.convolve(a, v, mode)``."""
    if n <= 0:
        raise ValueError("n must be a positive integer.")
    a = np.asarray(a)
    if a.size == 0:
        raise ValueError("len(a) must be at least 1.")
    if mode not in ("full", "valid", "same"):
        raise ValueError("'mode' argument must be one of ('full', 'valid', 'same')")
    if a.ndim > 1:
        return np.apply_along_axis(lambda a: convolution_matrix(a, n, mode), -1, a)
    az = np.pad(a, (0, n - 1), "constant")
    raz = np.pad(a[::-1], (0, n - 1), "constant")
    if mode == "same":
        trim = min(n, len(a)) - 1
        tb = trim // 2
        te = trim - tb
        col0 = az[tb : len(az) - te]
        row0 = raz[-n - tb : len(raz) - tb]
    elif mode == "valid":
        tb = min(n, len(a)) - 1
        te = tb
        col0 = az[tb : len(az) - te]
        row0 = raz[-n - tb : len(raz) - tb]
    else:
        col0 = az
        row0 = raz[-n:]
    return toeplitz(col0, row0)
