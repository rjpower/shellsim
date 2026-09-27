"""Linear systems, inverses, determinants and least squares, following SciPy 1.18's
``scipy/linalg/_basic.py``.

``solve``, ``inv`` and ``det`` call the native ``_scipy_linalg`` kernels that port SciPy's
batched C++ loops, so structure detection, condition estimates and the per-slice errors and
``LinAlgWarning`` messages match SciPy's. ``solve_triangular`` and ``solve_banded`` call the
f2py-style LAPACK wrappers, as SciPy does.

``lstsq`` has no native kernel: it solves each slice with ``numpy.linalg.lstsq``, which uses the
singular value decomposition, for every ``lapack_driver``. Solutions, ranks and singular values
agree with SciPy's up to rounding, and residuals follow SciPy's rules (NaN for a rank-deficient
slice, empty unless the system is overdetermined, empty for ``'gelsy'``). ``'gelsy'`` decides
the rank from the singular values rather than from a pivoted QR factorization.

``pinv`` and ``pinvh`` use ``svd`` and ``eigh``, which compute ``float32`` input in double
precision. ``solve_circulant`` uses ``numpy.fft``. Complex input, ``assume_a='banded'`` in
``solve``, and SciPy's other solvers raise ``NotImplementedError``.
"""

import warnings

import numpy as np
from _scipy_linalg import _det as _linalg_det
import _scipy_linalg as _batched_linalg
from scipy._lib._util import _apply_over_batch, _asarray_validated, _deprecate_dtypes
from scipy.linalg import _decomp, _decomp_svd
from scipy.linalg._misc import LinAlgError, LinAlgWarning, _datacopied, _reject_complex
from scipy.linalg.lapack import (
    _ensure_aligned_and_native,
    _ensure_dtype_cdsz,
    _normalize_lapack_dtype,
    _normalize_lapack_dtype1,
    get_lapack_funcs,
)

__all__ = [
    "solve",
    "solve_triangular",
    "solve_banded",
    "solve_circulant",
    "inv",
    "det",
    "lstsq",
    "pinv",
    "pinvh",
]


def _format_emit_errors_warnings(err_lst):
    """Raise or warn for the per-slice problems a batched kernel reports, as SciPy does.

    Singular slices raise ``LinAlgError``, LAPACK argument errors raise ``ValueError``, and
    ill-conditioned slices warn with ``LinAlgWarning``. Like SciPy, the messages number slices
    by their position in ``err_lst``.
    """
    singular, lapack_err, ill_cond = [], [], []
    for i, dct in enumerate(err_lst):
        if dct["is_singular"]:
            singular.append(i)
        if dct["lapack_info"] < 0:
            lapack_err.append(f"slice {i} emits lapack info={dct['lapack_info']}")
        if dct["is_ill_conditioned"]:
            ill_cond.append(f"slice {i} has rcond = {dct['rcond']}")
    if singular:
        raise LinAlgError(f"A singular matrix detected: slice(s) {singular} are singular.")
    if lapack_err:
        raise ValueError(f"Internal LAPACK errors: {','.join(lapack_err)}.")
    if ill_cond:
        warnings.warn(
            f"An ill-conditioned matrix detected: {','.join(ill_cond)}.",
            LinAlgWarning,
            stacklevel=3,
        )


def solve(
    a,
    b,
    lower=False,
    overwrite_a=False,
    overwrite_b=False,
    check_finite=True,
    assume_a=None,
    transposed=False,
):
    """Solve ``a @ x = b`` (or ``a.T @ x = b``) for square ``a``, batched over leading axes.

    Without ``assume_a`` the structure of each matrix is detected: diagonal, tridiagonal,
    triangular, symmetric positive definite, symmetric, or general.
    """
    structure = {
        None: -1,
        "general": 0,
        "gen": 0,
        "diagonal": 11,
        "tridiagonal": 31,
        "banded": 41,
        "upper triangular": 21,
        "lower triangular": 22,
        "pos": 101,
        "positive definite": 101,
        "sym": 201,
        "symmetric": 201,
        "her": 211,
        "hermitian": 211,
    }.get(assume_a, "unknown")
    if structure == "unknown":
        raise ValueError(f"{assume_a} is not a recognized matrix structure")

    a1 = np.atleast_2d(_asarray_validated(a, check_finite=check_finite))
    b1 = np.atleast_1d(_asarray_validated(b, check_finite=check_finite))
    _deprecate_dtypes("linalg.solve", a1, b1)

    a1, b1 = _ensure_dtype_cdsz(a1, b1)
    a1, overwrite_a = _normalize_lapack_dtype(a1, overwrite_a)
    a1, overwrite_a = _ensure_aligned_and_native(a1, overwrite_a)
    b1, overwrite_b = _ensure_aligned_and_native(b1, overwrite_b)

    if a1.ndim < 2:
        raise ValueError(f"Expected at least ndim=2, got {a1.ndim=}")
    if a1.shape[-1] != a1.shape[-2]:
        raise ValueError(f"Expected square matrix, got {a1.shape=}")

    if np.issubdtype(a1.dtype, np.complexfloating) and transposed:
        raise NotImplementedError(
            "scipy.linalg.solve can currently not solve a^T x = b or a^H x = b "
            "for complex matrices."
        )

    b_is_1D = b1.ndim == 1
    if b_is_1D:
        b1 = b1[:, None]

    a_is_scalar = a1.size == 1
    if b1.shape[-2] != a1.shape[-1] and not a_is_scalar:
        raise ValueError(f"incompatible shapes: {a1.shape=} and {b1.shape=}")

    batch_shape = np.broadcast_shapes(a1.shape[:-2], b1.shape[:-2])
    a1 = np.broadcast_to(a1, batch_shape + a1.shape[-2:])
    b1 = np.broadcast_to(b1, batch_shape + b1.shape[-2:])

    if a1.size == 0 or b1.size == 0:
        x = np.empty_like(b1)
        if b_is_1D:
            x = x[..., 0]
        return x

    if a_is_scalar:
        if a1.item() == 0:
            raise LinAlgError("A singular matrix detected.")
        out = b1 / a1
        return out[..., 0] if b_is_1D else out

    _reject_complex("solve", a1, b1)
    x, err_lst = _batched_linalg._solve(a1, b1, structure, lower, transposed)
    if err_lst:
        _format_emit_errors_warnings(err_lst)
    if b_is_1D:
        x = x[..., 0]
    return x


@_apply_over_batch(("a", 2), ("b", "1|2"))
def solve_triangular(
    a, b, trans=0, lower=False, unit_diagonal=False, overwrite_b=False, check_finite=True
):
    """Solve ``a @ x = b`` for triangular ``a``; ``trans`` 1 or ``'T'`` solves ``a.T @ x = b``."""
    a1 = _asarray_validated(a, check_finite=check_finite)
    b1 = _asarray_validated(b, check_finite=check_finite)
    if len(a1.shape) != 2 or a1.shape[0] != a1.shape[1]:
        raise ValueError("expected square matrix")
    if a1.shape[0] != b1.shape[0]:
        raise ValueError(f"shapes of a {a1.shape} and b {b1.shape} are incompatible")
    if b1.size == 0:
        dt_nonempty = solve_triangular(np.eye(2, dtype=a1.dtype), np.ones(2, dtype=b1.dtype)).dtype
        return np.empty_like(b1, dtype=dt_nonempty)
    overwrite_b = overwrite_b or _datacopied(b1, b)
    x, _ = _solve_triangular(a1, b1, trans, lower, unit_diagonal, overwrite_b)
    return x


def _solve_triangular(a1, b1, trans=0, lower=False, unit_diagonal=False, overwrite_b=False):
    """``solve_triangular`` without input validation: ``(x, info)``."""
    _reject_complex("solve_triangular", a1, b1)
    trans = {"N": 0, "T": 1, "C": 2}.get(trans, trans)
    (trtrs,) = get_lapack_funcs(("trtrs",), (a1, b1))
    if a1.flags.f_contiguous or trans == 2:
        x, info = trtrs(
            a1, b1, overwrite_b=overwrite_b, lower=lower, trans=trans, unitdiag=unit_diagonal
        )
    else:
        # SciPy solves the transposed system, since trtrs expects Fortran order.
        x, info = trtrs(
            a1.T,
            b1,
            overwrite_b=overwrite_b,
            lower=not lower,
            trans=not trans,
            unitdiag=unit_diagonal,
        )
    if info == 0:
        return x, info
    if info > 0:
        raise LinAlgError(f"singular matrix: resolution failed at diagonal {info - 1}")
    raise ValueError(f"illegal value in {-info}-th argument of internal trtrs")


def solve_banded(l_and_u, ab, b, overwrite_ab=False, overwrite_b=False, check_finite=True):
    """Solve ``a @ x = b`` for a band matrix ``a`` given in LAPACK's band storage ``ab``.

    ``l_and_u`` is ``(l, u)``, the numbers of nonzero lower and upper diagonals, and
    ``ab[u + i - j, j] == a[i, j]``.
    """
    (nlower, nupper) = l_and_u
    return _solve_banded(
        nlower,
        nupper,
        ab,
        b,
        overwrite_ab=overwrite_ab,
        overwrite_b=overwrite_b,
        check_finite=check_finite,
    )


@_apply_over_batch(("nlower", 0), ("nupper", 0), ("ab", 2), ("b", "1|2"))
def _solve_banded(nlower, nupper, ab, b, overwrite_ab, overwrite_b, check_finite):
    a1 = _asarray_validated(ab, check_finite=check_finite, as_inexact=True)
    b1 = _asarray_validated(b, check_finite=check_finite, as_inexact=True)
    if a1.shape[-1] != b1.shape[0]:
        raise ValueError("shapes of ab and b are not compatible.")
    if nlower + nupper + 1 != a1.shape[0]:
        raise ValueError(
            f"invalid values for the number of lower and upper diagonals: l+u+1 "
            f"({nlower + nupper + 1}) does not equal ab.shape[0] ({ab.shape[0]})"
        )
    if b1.size == 0:
        dt = solve(np.eye(1, dtype=a1.dtype), np.ones(1, dtype=b1.dtype)).dtype
        return np.empty_like(b1, dtype=dt)
    overwrite_b = overwrite_b or _datacopied(b1, b)
    if a1.shape[-1] == 1:
        b2 = np.array(b1, copy=(not overwrite_b))
        # For a 1x1 matrix the diagonal is in row `u` of `ab` (gh-8906).
        b2 /= a1[nupper, 0]
        return b2
    _reject_complex("solve_banded", a1, b1)
    if nlower == nupper == 1:
        overwrite_ab = overwrite_ab or _datacopied(a1, ab)
        (gtsv,) = get_lapack_funcs(("gtsv",), (a1, b1))
        du = a1[0, 1:]
        d = a1[1, :]
        dl = a1[2, :-1]
        du2, d, du, x, info = gtsv(
            dl, d, du, b1, overwrite_ab, overwrite_ab, overwrite_ab, overwrite_b
        )
    else:
        (gbsv,) = get_lapack_funcs(("gbsv",), (a1, b1))
        a2 = np.zeros((2 * nlower + nupper + 1, a1.shape[1]), dtype=gbsv.dtype)
        a2[nlower:, :] = a1
        lu, piv, x, info = gbsv(nlower, nupper, a2, b1, overwrite_ab=True, overwrite_b=overwrite_b)
    if info == 0:
        return x
    if info > 0:
        raise LinAlgError("singular matrix")
    raise ValueError(f"illegal value in {-info}-th argument of internal gbsv/gtsv")


def _get_axis_len(aname, a, axis):
    ax = axis
    if ax < 0:
        ax += a.ndim
    if 0 <= ax < a.ndim:
        return a.shape[ax]
    raise ValueError(f"'{aname}axis' entry is out of bounds")


def solve_circulant(c, b, singular="raise", tol=None, caxis=-1, baxis=0, outaxis=0):
    """Solve ``C @ x = b`` for the circulant matrix ``C`` whose first column is ``c``, using
    the discrete Fourier transform."""
    c = np.atleast_1d(c)
    nc = _get_axis_len("c", c, caxis)
    b = np.atleast_1d(b)
    nb = _get_axis_len("b", b, baxis)
    if nc != nb:
        raise ValueError(f"Shapes of c {c.shape} and b {b.shape} are incompatible")
    _deprecate_dtypes("solve_circulant", c, b)
    if b.size == 0:
        dt = solve_circulant(np.arange(3, dtype=c.dtype), np.ones(3, dtype=b.dtype)).dtype
        return np.empty_like(b, dtype=dt)

    fc = np.fft.fft(np.moveaxis(c, caxis, -1), axis=-1)
    abs_fc = np.abs(fc)
    if tol is None:
        # The tolerance np.linalg.matrix_rank uses.
        tol = abs_fc.max(axis=-1) * nc * np.finfo(np.float64).eps
        if tol.shape != ():
            tol = tol.reshape(tol.shape + (1,))
        else:
            tol = np.atleast_1d(tol)

    near_zeros = abs_fc <= tol
    is_near_singular = np.any(near_zeros)
    if is_near_singular:
        if singular == "raise":
            raise LinAlgError("near singular circulant matrix.")
        # Avoid dividing by the near-zero values; their quotients are zeroed below.
        fc[near_zeros] = 1

    fb = np.fft.fft(np.moveaxis(b, baxis, -1), axis=-1)
    q = fb / fc
    if is_near_singular:
        mask = np.ones_like(b, dtype=bool) & near_zeros
        q[mask] = 0

    x = np.fft.ifft(q, axis=-1)
    if not (np.iscomplexobj(c) or np.iscomplexobj(b)):
        x = x.real
    if outaxis != -1:
        x = np.moveaxis(x, -1, outaxis)
    return x


def inv(a, overwrite_a=False, check_finite=True, *, assume_a=None, lower=False):
    """The inverse of a square matrix, or of each matrix in a stack.

    Without ``assume_a`` the structure is detected: diagonal, triangular, symmetric positive
    definite, symmetric, or general.
    """
    a1 = _asarray_validated(a, check_finite=check_finite)
    _deprecate_dtypes("linalg.inv", a1)
    if a1.ndim < 2:
        raise ValueError(f"Expected at least ndim=2, got {a1.ndim=}")
    if a1.shape[-1] != a1.shape[-2]:
        raise ValueError(f"Expected square matrix, got {a1.shape=}")
    if a1.size == 0:
        dt = inv(np.eye(2, dtype=a1.dtype)).dtype
        return np.empty_like(a1, dtype=dt)
    a1, overwrite_a = _normalize_lapack_dtype(a1, overwrite_a)
    a1, overwrite_a = _ensure_aligned_and_native(a1, overwrite_a)
    structure = {
        None: -1,
        "general": 0,
        "gen": 0,
        "diagonal": 11,
        "upper triangular": 21,
        "lower triangular": 22,
        "pos": 101,
        "sym": 201,
        "her": 211,
    }[assume_a]
    _reject_complex("inv", a1)
    inv_a, err_lst = _batched_linalg._inv(a1, structure, lower)
    if err_lst:
        _format_emit_errors_warnings(err_lst)
    return inv_a


def det(a, overwrite_a=False, check_finite=True):
    """The determinant of a square matrix, or of each matrix in a stack.

    ``float32`` results are returned as ``float64``, and a 2-d input gives a NumPy scalar.
    """
    a1 = np.asarray_chkfinite(a) if check_finite else np.asarray(a)
    _deprecate_dtypes("linalg.det", a1)
    if a1.ndim < 2:
        raise ValueError("The input array must be at least two-dimensional.")
    if a1.shape[-1] != a1.shape[-2]:
        raise ValueError(
            f"Last 2 dimensions of the array must be square but received shape {a1.shape}."
        )
    a1, overwrite_a = _normalize_lapack_dtype1(a1, overwrite_a)
    if min(*a1.shape) == 0:
        dtyp = np.float64 if a1.dtype.char not in "FD" else np.complex128
        if a1.ndim == 2:
            return dtyp(1.0)
        return np.ones(shape=a1.shape[:-2], dtype=dtyp)
    if a1.shape[-2:] == (1, 1):
        a1 = a1[..., 0, 0]
        if a1.ndim == 0:
            a1 = a1[()]
        if a1.dtype.char in "dD":
            return a1
        return a1.astype("d") if a1.dtype.char == "f" else a1.astype("D")

    _reject_complex("det", a1)
    det = _linalg_det(a1)
    # Promote single precision to double to prevent overflows.
    if det.dtype.char == "f":
        det = det.astype(np.float64)
    if det.ndim == 0:
        return det[()]
    return det


def lstsq(
    a, b, cond=None, overwrite_a=False, overwrite_b=False, check_finite=True, lapack_driver=None
):
    """The least-squares solution of ``a @ x = b``: ``(x, residues, rank, s)``.

    ``cond`` is the relative cutoff below which singular values count as zero; it defaults to
    the machine epsilon of ``a``'s precision. ``s`` is ``None`` for the ``'gelsy'`` driver.
    """
    driver = lapack_driver
    if driver is None:
        driver = _DEFAULT_LAPACK_DRIVER
    if driver not in ("gelsd", "gelsy", "gelss"):
        raise ValueError(f'LAPACK driver "{driver}" is not found')
    if len(a.shape) < 2:
        raise ValueError("Input array a should be at least 2D, got {a.shape = }")

    a1 = np.atleast_2d(_asarray_validated(a, check_finite=check_finite))
    b1 = np.atleast_1d(_asarray_validated(b, check_finite=check_finite))
    _deprecate_dtypes("linalg.lstsq", a1, b1)
    a1, b1 = _ensure_dtype_cdsz(a1, b1)
    a1, overwrite_a = _normalize_lapack_dtype(a1, overwrite_a)
    a1, overwrite_a = _ensure_aligned_and_native(a1, overwrite_a)
    b1, overwrite_b = _ensure_aligned_and_native(b1, overwrite_b)

    m, n = a1.shape[-2:]
    if m == 0 or n == 0:
        x = np.zeros((n,) + b1.shape[1:], dtype=np.common_type(a1, b1))
        if n == 0:
            residues = np.linalg.norm(b1, axis=0) ** 2
        else:
            residues = np.empty((0,))
        return x, residues, 0, np.empty((0,))

    b_is_1D = b1.ndim == 1
    if b_is_1D:
        b1 = b1[:, None]
    if m != b1.shape[-2]:
        raise ValueError(
            f"Shape mismatch: a and b should have the same number of rows ({m} != {b1.shape[-2]})."
        )
    batch_shape = np.broadcast_shapes(a1.shape[:-2], b1.shape[:-2])
    a1 = np.broadcast_to(a1, batch_shape + a1.shape[-2:])
    b1 = np.broadcast_to(b1, batch_shape + b1.shape[-2:])

    if cond is None:
        cond = np.finfo(a1.dtype).eps
    else:
        cond = float(cond)

    _reject_complex("lstsq", a1, b1)
    nrhs = b1.shape[-1]
    x1 = np.empty(batch_shape + (n, nrhs), dtype=a1.dtype)
    rank = np.empty(batch_shape, dtype=np.int64)
    S = np.empty(batch_shape + (min(m, n),), dtype=a1.dtype)
    residuals = np.empty(batch_shape + (nrhs,), dtype=a1.dtype)
    for index in np.ndindex(batch_shape):
        a_slice, b_slice = a1[index], b1[index]
        x, _, rank[index], S[index] = np.linalg.lstsq(a_slice, b_slice, rcond=cond)
        x1[index] = x
        if m > n:
            r = b_slice - a_slice @ x
            residuals[index] = np.sum(r * r, axis=-2)

    if m > n and lapack_driver != "gelsy":
        # LAPACK makes no promises about residuals without full column rank.
        residuals[rank < n, :] = np.nan
    else:
        residuals = np.zeros(batch_shape + (0,), dtype=a1.dtype)
    if b_is_1D:
        x1 = x1[..., 0]
        if residuals.size > 0 and lapack_driver != "gelsy":
            residuals = residuals[..., 0]
    if m <= n:
        residuals = np.zeros(batch_shape + (0,), dtype=residuals.dtype)
    if driver == "gelsy":
        S = None
    if rank.ndim == 0:
        rank = rank[()]
    return x1, residuals, rank, S


# SciPy keeps this in `lstsq.default_lapack_driver`; shellsim's functions take no attributes.
_DEFAULT_LAPACK_DRIVER = "gelsd"


def pinv(a, *, atol=None, rtol=None, return_rank=False, check_finite=True):
    """The Moore-Penrose pseudo-inverse, from the singular value decomposition.

    Singular values at or below ``atol + rtol * max(s)`` are treated as zero; ``rtol``
    defaults to ``max(M, N) * eps``.
    """
    a = _asarray_validated(a, check_finite=check_finite)
    u, s, vh = _decomp_svd.svd(a.conj(), full_matrices=False, check_finite=False)
    atol = 0.0 if atol is None else atol
    rtol = max(a.shape[-2:]) * np.finfo(u.dtype).eps if (rtol is None) else rtol
    if (atol < 0.0) or (rtol < 0.0):
        raise ValueError("atol and rtol values must be positive.")
    maxS = np.max(s, axis=-1, initial=0.0, keepdims=True)
    val = atol + maxS * rtol
    large = s > val
    rank = np.sum(large, axis=-1)
    np.divide(1, s, where=large, out=s)
    s[~large] = 0
    B = vh.mT @ (s[..., None] * u.mT)
    if return_rank:
        return B, rank
    return B


@_apply_over_batch(("a", 2))
def pinvh(a, atol=None, rtol=None, lower=True, return_rank=False, check_finite=True):
    """The pseudo-inverse of a symmetric matrix, from its eigendecomposition."""
    a = _asarray_validated(a, check_finite=check_finite)
    s, u = _decomp.eigh(a, lower=lower, check_finite=False, driver="ev")
    t = u.dtype.char.lower()
    maxS = np.max(np.abs(s), initial=0.0)
    atol = 0.0 if atol is None else atol
    rtol = max(a.shape) * np.finfo(t).eps if (rtol is None) else rtol
    if (atol < 0.0) or (rtol < 0.0):
        raise ValueError("atol and rtol values must be positive.")
    val = atol + maxS * rtol
    above_cutoff = abs(s) > val
    psigma_diag = 1.0 / s[above_cutoff]
    u = u[:, above_cutoff]
    B = (u * psigma_diag) @ u.conj().T
    if return_rank:
        return B, len(psigma_diag)
    return B
