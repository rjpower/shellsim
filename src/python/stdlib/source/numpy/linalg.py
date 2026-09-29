"""shellsim's ``numpy.linalg``.

Real matrices use the metered native dense kernels. Complex operations compose from those
kernels and NumPy arrays: block-real solves and eigensolvers, direct elimination and Cholesky,
Householder QR, and a one-sided Jacobi SVD. These small, readable paths trade speed for
coverage of ordinary scientific calculations. Array operations and interpreter loops meter
their work; no host numerical library is called. ``_numpy_linalg`` also exposes primitives for
``scipy.linalg``. See ``docs/numpy.md`` for the supported frontier.
"""

import numpy as np
from _numpy import _LinAlgError as LinAlgError
from _numpy_linalg import cholesky as _cholesky
from _numpy_linalg import det as _det
from _numpy_linalg import eig as _eig
from _numpy_linalg import eigh as _eigh
from _numpy_linalg import inv as _inv
from _numpy_linalg import qr as _qr
from _numpy_linalg import slogdet as _slogdet
from _numpy_linalg import solve as _solve
from _numpy_linalg import svd as _svd

__all__ = [
    "LinAlgError",
    "inv",
    "solve",
    "det",
    "slogdet",
    "cholesky",
    "qr",
    "eigh",
    "eigvalsh",
    "eig",
    "eigvals",
    "svd",
    "svdvals",
    "matrix_power",
    "matrix_rank",
    "pinv",
    "lstsq",
    "norm",
]


def inv(a):
    """The multiplicative inverse of a square matrix, or a stack of them."""
    if np.iscomplexobj(a):
        a = np.asarray(a)
        n, _ = _matrix_shape(a, square=True)
        inverse = _inv(_real_block(a))
        return (inverse[..., :n, :n] + 1j * inverse[..., n:, :n]).astype(a.dtype)
    return _inv(a)


def solve(a, b):
    """The solution ``x`` of ``a @ x == b``."""
    a = np.asarray(a)
    b = np.asarray(b)
    work_dtype = np.complex128 if np.iscomplexobj(a) or np.iscomplexobj(b) else np.float64
    work_a = a.astype(work_dtype)
    work_b = b.astype(work_dtype)
    result = _solve_once(work_a, work_b)
    if result.size:
        vector = b.ndim == 1
        for _ in range(2):
            residual = work_b - _solve_product(work_a, result, vector)
            residual_size = np.max(np.abs(residual))
            if residual_size == 0:
                break
            correction = _solve_once(work_a, residual)
            candidate = result + correction
            next_residual = work_b - _solve_product(work_a, candidate, vector)
            if np.max(np.abs(next_residual)) >= residual_size:
                break
            result = candidate
    return result.astype(np.result_type(a, b, np.float32))


def _solve_product(a, x, vector):
    if vector:
        return (a @ x[..., np.newaxis])[..., 0]
    return a @ x


def _solve_once(a, b):
    """Use the real kernel for one solve, expanding complex arithmetic into a real block."""
    if np.iscomplexobj(a) or np.iscomplexobj(b):
        n, _ = _matrix_shape(a, square=True)
        vector = b.ndim == 1
        if vector:
            rhs = np.concatenate((b.real, b.imag), axis=-1)
        else:
            rhs = np.concatenate((b.real, b.imag), axis=-2)
        result = _solve(_real_block(a), rhs)
        if vector:
            value = result[..., :n] + 1j * result[..., n:]
        else:
            value = result[..., :n, :] + 1j * result[..., n:, :]
        return value
    return _solve(a, b)


def _real_block(a):
    """Represent a complex matrix as a real linear map on stacked real and imaginary parts."""
    a = np.asarray(a)
    top = np.concatenate((a.real, -a.imag), axis=-1)
    bottom = np.concatenate((a.imag, a.real), axis=-1)
    return np.concatenate((top, bottom), axis=-2)


def _matrix_shape(a, square=False):
    if a.ndim < 2:
        raise LinAlgError("array must be at least two-dimensional")
    rows, cols = a.shape[-2:]
    if square and rows != cols:
        raise LinAlgError("matrix must be square")
    return rows, cols


def det(a):
    """The determinant of a square matrix, or a stack of them."""
    if np.iscomplexobj(a):
        sign, logabs = slogdet(a)
        return sign * np.exp(logabs)
    return _det(a)


def slogdet(a):
    """The sign and the natural log of the absolute value of the determinant."""
    if np.iscomplexobj(a):
        a = np.asarray(a)
        n, _ = _matrix_shape(a, square=True)
        signs = []
        logs = []
        for source in a.reshape((-1, n, n)):
            matrix = source.astype(np.complex128).copy()
            sign = 1 + 0j
            logabs = 0.0
            for col in range(n):
                pivot_row = col + int(np.argmax(np.abs(matrix[col:, col])))
                pivot = matrix[pivot_row, col]
                if pivot == 0:
                    sign = 0j
                    logabs = -np.inf
                    break
                if pivot_row != col:
                    saved = matrix[col].copy()
                    matrix[col] = matrix[pivot_row]
                    matrix[pivot_row] = saved
                    sign = -sign
                sign *= pivot / abs(pivot)
                logabs += np.log(abs(pivot))
                for row in range(col + 1, n):
                    factor = matrix[row, col] / pivot
                    for trailing in range(col + 1, n):
                        matrix[row, trailing] -= factor * matrix[col, trailing]
            signs.append(sign)
            logs.append(logabs)
        sign_array = np.array(signs, dtype=a.dtype).reshape(a.shape[:-2])
        log_dtype = np.float32 if a.dtype == np.complex64 else np.float64
        log_array = np.array(logs, dtype=log_dtype).reshape(a.shape[:-2])
        return sign_array[()] if a.ndim == 2 else sign_array, log_array[()] if a.ndim == 2 else log_array
    return _slogdet(a)


def cholesky(a):
    """The lower-triangular factor ``L`` with ``L @ L.conj().T == a``."""
    if np.iscomplexobj(a):
        a = np.asarray(a)
        n, _ = _matrix_shape(a, square=True)
        factors = []
        for source in a.reshape((-1, n, n)):
            lower = np.zeros((n, n), dtype=np.complex128)
            for row in range(n):
                for col in range(row + 1):
                    value = source[row, col]
                    for k in range(col):
                        value -= lower[row, k] * lower[col, k].conjugate()
                    if row == col:
                        if value.real <= 0:
                            raise LinAlgError("matrix is not positive definite")
                        lower[row, col] = np.sqrt(value.real)
                    else:
                        lower[row, col] = value / lower[col, col]
            factors.append(lower)
        return np.array(factors, dtype=a.dtype).reshape(a.shape)
    return _cholesky(a)


def qr(a, mode="reduced"):
    """The QR factorization of a matrix, or a stack of them.

    ``mode`` follows real NumPy: ``"reduced"`` (default), ``"complete"``, ``"r"`` (``R`` only),
    or ``"raw"`` (the packed Householder reflectors and their scale factors, as LAPACK's
    ``geqrf`` leaves them).
    """
    if np.iscomplexobj(a):
        return _complex_qr(a, mode)
    return _qr(a, mode)


def _orthogonalize(vector, basis):
    """Two passes complete an orthonormal basis at zero singular values."""
    vector = vector.copy()
    for _ in range(2):
        for column in basis:
            vector -= np.vdot(column, vector) * column
    return vector


def _complete_basis(basis, size, count):
    result = list(basis)
    for seed in range(size):
        if len(result) == count:
            break
        candidate = np.zeros(size, dtype=np.complex128)
        candidate[seed] = 1
        candidate = _orthogonalize(candidate, result)
        length = np.sqrt(np.vdot(candidate, candidate).real)
        if length > 1e-10:
            result.append(candidate / length)
    return result


def _complex_qr(a, mode):
    """Factor a complex matrix with unitary Householder reflections."""
    if mode not in ("reduced", "complete", "r"):
        if mode == "raw":
            raise NotImplementedError("raw complex QR is not supported")
        raise ValueError("unrecognized QR mode")
    a = np.asarray(a)
    rows, cols = _matrix_shape(a)
    reduced = min(rows, cols)
    batch_shape = a.shape[:-2]
    q_results = []
    r_results = []
    for source in a.reshape((-1, rows, cols)):
        r = source.astype(np.complex128).copy()
        q = np.eye(rows, dtype=np.complex128)
        for col in range(reduced):
            column = r[col:, col].copy()
            length = _scaled_norm(column, axis=None, keepdims=False)
            if length == 0:
                continue
            phase = column[0] / abs(column[0]) if column[0] != 0 else 1
            column[0] += phase * length
            column /= _scaled_norm(column, axis=None, keepdims=False)
            trailing = r[col:, col:]
            r[col:, col:] = trailing - 2 * np.outer(column, column.conj() @ trailing)
            trailing_q = q[:, col:]
            q[:, col:] = trailing_q - 2 * np.outer(trailing_q @ column, column.conj())
            r[col + 1 :, col] = 0
        if mode == "complete":
            q_results.append(q)
            r_results.append(r)
        else:
            q_results.append(q[:, :reduced])
            r_results.append(r[:reduced])
    r_rows = rows if mode == "complete" else reduced
    result_r = np.array(r_results, dtype=a.dtype).reshape(batch_shape + (r_rows, cols))
    if mode == "r":
        return result_r
    q_cols = rows if mode == "complete" else reduced
    result_q = np.array(q_results, dtype=a.dtype).reshape(batch_shape + (rows, q_cols))
    return result_q, result_r


def eigh(a, UPLO="L"):
    """Eigenvalues (ascending) and orthonormal eigenvectors of a Hermitian matrix.

    Only the triangle named by ``UPLO`` ("L" for lower, the default, or "U" for upper) is read;
    the other is assumed to mirror it. Eigenvector signs are normalized so each column's
    largest-magnitude entry is positive (ties keep the first such entry positive); real NumPy's
    signs come from LAPACK and are not otherwise comparable.
    """
    if np.iscomplexobj(a):
        return _complex_eigh(a, UPLO, True)
    return _eigh(a, UPLO, True)


def eigvalsh(a, UPLO="L"):
    """Eigenvalues (ascending) of a Hermitian matrix, without eigenvectors."""
    if np.iscomplexobj(a):
        w, _ = _complex_eigh(a, UPLO, False)
        return w
    w, _ = _eigh(a, UPLO, False)
    return w


def _complex_eigh(a, UPLO, vectors):
    """Use the real symmetric 2n representation, keeping one complex basis per eigenspace."""
    a = np.asarray(a)
    if UPLO not in ("L", "U"):
        raise ValueError("UPLO must be L or U")
    n, _ = _matrix_shape(a, square=True)
    batch_shape = a.shape[:-2]
    values = []
    bases = []
    for source in a.reshape((-1, n, n)):
        hermitian = np.zeros((n, n), dtype=np.complex128)
        for row in range(n):
            hermitian[row, row] = source[row, row].real
            for col in range(row):
                entry = source[row, col] if UPLO == "L" else source[col, row].conjugate()
                hermitian[row, col] = entry
                hermitian[col, row] = entry.conjugate()
        real_values, real_vectors = _eigh(_real_block(hermitian), "L", vectors)
        if not vectors:
            values.append(real_values[::2])
            continue
        chosen_values = []
        chosen_vectors = []
        for index in range(2 * n):
            candidate = real_vectors[:n, index] + 1j * real_vectors[n:, index]
            for previous in chosen_vectors:
                candidate = candidate - np.vdot(previous, candidate) * previous
            length = np.sqrt(np.vdot(candidate, candidate).real)
            if length < 1e-8:
                continue
            candidate = candidate / length
            pivot = int(np.argmax(np.abs(candidate)))
            candidate = candidate * (candidate[pivot].conjugate() / abs(candidate[pivot]))
            chosen_values.append(real_values[index])
            chosen_vectors.append(candidate)
            if len(chosen_values) == n:
                break
        if len(chosen_values) != n:
            raise LinAlgError("eigenvectors did not span the complex space")
        values.append(chosen_values)
        bases.append(np.array(chosen_vectors).T)
    real_dtype = np.float32 if a.dtype == np.complex64 else np.float64
    result_values = np.array(values, dtype=real_dtype).reshape(batch_shape + (n,))
    if not vectors:
        return result_values, None
    result_vectors = np.array(bases, dtype=a.dtype).reshape(batch_shape + (n, n))
    return result_values, result_vectors


def eig(a):
    """Eigenvalues and right eigenvectors of a general square matrix, or a stack of them.

    Returns ``(w, v)``: `w` holds the eigenvalues (not necessarily ordered, and not necessarily
    real even when `a` is real: a real matrix's complex eigenvalues occur in conjugate pairs),
    and column ``v[:, i]`` is the eigenvector for ``w[i]``, normalized to unit length. `w` and `v`
    are always complex, matching real NumPy's own `eig` (whose docs describe casting an
    all-real result down to a real dtype, but whose current implementation, like this one, does
    not). Eigenvector signs (more generally, for a complex eigenvector, its phase) are normalized
    so each column's largest-magnitude entry is a positive real number; real NumPy's come from
    LAPACK and are not otherwise comparable.

    Raises :class:`LinAlgError` if the underlying QR iteration fails to converge, or if `a` is
    not square.
    """
    if np.iscomplexobj(a):
        return _complex_eig(a)
    return _eig(a, True)


def eigvals(a):
    """Eigenvalues of a general square matrix, or a stack of them, without eigenvectors. See
    :func:`eig`."""
    if np.iscomplexobj(a):
        w, _ = _complex_eig(a)
        return w
    w, _ = _eig(a, False)
    return w


def _complex_eig(a):
    """Recover A's eigenpairs from its real 2n representation's complex eigenpairs."""
    a = np.asarray(a)
    n, _ = _matrix_shape(a, square=True)
    values = []
    vectors = []
    for source in a.reshape((-1, n, n)):
        real_values, real_vectors = _eig(_real_block(source), True)
        chosen_values = []
        chosen_vectors = []
        groups = []
        for index, value in enumerate(real_values):
            for group in groups:
                representative = real_values[group[0]]
                if abs(value - representative) < 1e-7 * (1 + abs(value)):
                    group.append(index)
                    break
            else:
                groups.append([index])
        for group in groups:
            value = real_values[group[0]]
            candidates = []
            for index in group:
                candidate = real_vectors[:n, index] + 1j * real_vectors[n:, index]
                length = np.sqrt(np.vdot(candidate, candidate).real)
                if length > 1e-8:
                    candidates.append(candidate / length)
            real_value = abs(value.imag) < 1e-8
            count = len(group) // 2 if real_value else len(candidates)
            independent = []
            for candidate in candidates:
                if len(independent) == count:
                    break
                if all(abs(np.vdot(previous, candidate)) < 1 - 1e-7 for previous in independent):
                    independent.append(candidate)
            if not independent and count:
                raise LinAlgError("eigenvectors did not span the complex space")
            if real_value:
                while len(independent) < count:
                    independent.append(independent[-1])
            for candidate in independent:
                pivot = int(np.argmax(np.abs(candidate)))
                candidate = candidate * (candidate[pivot].conjugate() / abs(candidate[pivot]))
                chosen_values.append(value)
                chosen_vectors.append(candidate)
        if not chosen_values:
            raise LinAlgError("eigenvectors did not span the complex space")
        if len(chosen_values) != n:
            raise LinAlgError("eigenvalue multiplicities did not match matrix size")
        values.append(chosen_values)
        vectors.append(np.array(chosen_vectors).T)
    result_values = np.array(values, dtype=a.dtype).reshape(a.shape[:-2] + (n,))
    result_vectors = np.array(vectors, dtype=a.dtype).reshape(a.shape)
    return result_values, result_vectors


def svd(a, full_matrices=True, compute_uv=True):
    """The singular value decomposition of a matrix, or a stack of them.

    Singular values are descending. When `compute_uv` is true, returns ``(u, s, vh)`` with
    ``u @ diag(s) @ vh`` reconstructing `a` (restricted to the leading `k = min(rows, cols)`
    columns of `u` and rows of `vh` when ``full_matrices`` is false). Singular-vector signs are
    normalized the same way :func:`eigh`'s eigenvector signs are.
    """
    if np.iscomplexobj(a):
        return _complex_svd(a, full_matrices, compute_uv)
    return _svd(a, full_matrices, compute_uv)


def svdvals(a):
    """The singular values of a matrix, or a stack of them, in descending order."""
    if np.iscomplexobj(a):
        return _complex_svd(a, True, False)
    return _svd(a, True, False)


def _complex_svd(a, full_matrices, compute_uv):
    """Orthogonalize column pairs without forming the ill-conditioned AᴴA."""
    a = np.asarray(a)
    rows, cols = _matrix_shape(a)
    count = min(rows, cols)
    batch_shape = a.shape[:-2]
    singular_values = []
    left_matrices = []
    right_matrices = []
    for source in a.reshape((-1, rows, cols)):
        columns = source.astype(np.complex128).copy()
        right = np.eye(cols, dtype=np.complex128)
        rank_floor = _scaled_norm(columns, axis=None, keepdims=False) * max(rows, cols) * np.finfo(np.float64).eps
        for _ in range(20 * cols):
            changed = False
            for first in range(cols):
                for second in range(first + 1, cols):
                    left_column = columns[:, first].copy()
                    right_column = columns[:, second].copy()
                    left_length = _scaled_norm(left_column, axis=None, keepdims=False)
                    right_length = _scaled_norm(right_column, axis=None, keepdims=False)
                    if left_length <= rank_floor or right_length <= rank_floor:
                        continue
                    cosine = np.vdot(left_column / left_length, right_column / right_length)
                    if abs(cosine) <= 1e-14:
                        continue
                    scale = max(left_length, right_length)
                    left_size = left_length / scale
                    right_size = right_length / scale
                    overlap = abs(cosine) * left_size * right_size
                    if overlap == 0:
                        continue
                    difference = (right_size - left_size) * (right_size + left_size)
                    half_difference = difference / 2
                    tangent = (1 if difference >= 0 else -1) * overlap / (
                        abs(half_difference) + np.sqrt(half_difference ** 2 + overlap ** 2)
                    )
                    if tangent == 0:
                        continue
                    changed = True
                    cosine_angle = 1 / np.sqrt(1 + tangent * tangent)
                    sine_angle = tangent * cosine_angle
                    phase = cosine.conjugate() / abs(cosine)
                    columns[:, first] = cosine_angle * left_column - sine_angle * phase * right_column
                    columns[:, second] = sine_angle * left_column + cosine_angle * phase * right_column
                    left_vector = right[:, first].copy()
                    right_vector = right[:, second].copy()
                    right[:, first] = cosine_angle * left_vector - sine_angle * phase * right_vector
                    right[:, second] = sine_angle * left_vector + cosine_angle * phase * right_vector
            if not changed:
                break
        else:
            raise LinAlgError("SVD did not converge")
        lengths = [_scaled_norm(columns[:, index], axis=None, keepdims=False) for index in range(cols)]
        order = sorted(range(cols), key=lambda index: lengths[index], reverse=True)
        values = [lengths[index] for index in order[:count]]
        singular_values.append(values)
        if not compute_uv:
            continue
        left = []
        tolerance = (values[0] if values else 0.0) * max(rows, cols) * np.finfo(np.float64).eps
        for index, value in enumerate(values):
            if value > tolerance:
                candidate = columns[:, order[index]] / value
                left.append(candidate)
                continue
            left = _complete_basis(left, rows, len(left) + 1)
        if full_matrices:
            left = _complete_basis(left, rows, rows)
        left_matrices.append(np.array(left).T)
        right_count = cols if full_matrices else count
        right_matrices.append(right[:, order[:right_count]].conj().T)
    real_dtype = np.float32 if a.dtype == np.complex64 else np.float64
    s = np.array(singular_values, dtype=real_dtype).reshape(batch_shape + (count,))
    if not compute_uv:
        return s
    left_cols = rows if full_matrices else count
    right_rows = cols if full_matrices else count
    u = np.array(left_matrices, dtype=a.dtype).reshape(batch_shape + (rows, left_cols))
    vh = np.array(right_matrices, dtype=a.dtype).reshape(batch_shape + (right_rows, cols))
    return u, s, vh


def matrix_power(a, n):
    """``a`` raised to the integer power `n` by repeated squaring; `n < 0` uses :func:`inv`."""
    a = np.asarray(a)
    n = int(n)
    if n == 0:
        return np.broadcast_to(np.eye(a.shape[-1], dtype=a.dtype), a.shape).copy()
    if n < 0:
        a = inv(a)
        n = -n
    result = None
    base = a
    while n > 0:
        if n & 1:
            result = base if result is None else result @ base
        n >>= 1
        if n:
            base = base @ base
    return result


def matrix_rank(a, tol=None, hermitian=False):
    """The number of singular values of `a` above a shape- and magnitude-derived tolerance.

    If `hermitian`, the (real) eigenvalues' magnitudes are used instead of the singular values,
    which is cheaper and exact for a genuinely Hermitian `a`.
    """
    a = np.asarray(a)
    if a.ndim < 2:
        return int(np.any(a))
    if hermitian:
        s = np.sort(np.abs(eigvalsh(a)), axis=-1)[..., ::-1]
    else:
        s = svdvals(a)
    if tol is None:
        largest = np.amax(s, axis=-1, keepdims=True)
        tol = largest * max(a.shape[-2:]) * np.finfo(s.dtype).eps
    return np.sum(s > tol, axis=-1)


def pinv(a, rcond=1e-15, hermitian=False):
    """The Moore-Penrose pseudo-inverse of `a`, computed from its SVD."""
    a = np.asarray(a)
    if not np.iscomplexobj(a) and a.dtype != np.float32:
        a = a.astype(np.float64)
    u, s, vt = svd(a, full_matrices=False)
    cutoff = rcond * np.amax(s, axis=-1, keepdims=True)
    large = s > cutoff
    safe = np.where(large, s, 1.0)
    s_inv = np.where(large, 1.0 / safe, 0.0)
    v = np.swapaxes(vt.conj(), -1, -2)
    ut = np.swapaxes(u.conj(), -1, -2)
    return (v * s_inv[..., np.newaxis, :]) @ ut


def lstsq(a, b, rcond=None):
    """The least-squares solution of ``a @ x == b``, via `a`'s SVD.

    Returns ``(x, residuals, rank, singular_values)``. `residuals` holds the sum of squared
    residuals per right-hand-side column when `a` has more rows than columns and full column
    rank; otherwise it is an empty array, as real NumPy's does.
    """
    a = np.asarray(a)
    if not np.iscomplexobj(a) and a.dtype != np.float32:
        a = a.astype(np.float64)
    b = np.asarray(b)
    m, n = a.shape
    is_1d = b.ndim == 1
    b2 = b[:, np.newaxis] if is_1d else b
    u, s, vt = svd(a, full_matrices=False)
    if rcond is None:
        rcond = np.finfo(s.dtype).eps * max(m, n)
    cutoff = rcond * s[0] if s.size else 0.0
    large = s > cutoff
    rank = int(np.sum(large))
    safe = np.where(large, s, 1.0)
    s_inv = np.where(large, 1.0 / safe, 0.0)
    utb = np.swapaxes(u.conj(), -1, -2) @ b2
    x = np.swapaxes(vt.conj(), -1, -2) @ (s_inv[:, np.newaxis] * utb)
    if m > n and rank == n:
        residuals = np.sum(np.abs(b2 - a @ x) ** 2, axis=0)
    else:
        residuals = np.empty(0)
    if is_1d:
        x = x[:, 0]
    return x, residuals, rank, s


def _scaled_norm(x, axis, keepdims):
    """Scale before squaring so finite two-norms survive extreme magnitudes."""
    magnitude = np.abs(x)
    if magnitude.size == 0:
        return np.sqrt(np.sum(magnitude ** 2, axis=axis, keepdims=keepdims))
    scale = np.max(magnitude, axis=axis, keepdims=True)
    safe_scale = np.where(scale == 0, 1, scale)
    squares = np.sum((magnitude / safe_scale) ** 2, axis=axis, keepdims=True)
    result = scale * np.sqrt(squares)
    result = np.where(np.isinf(scale), scale, result)
    if keepdims:
        return result
    return np.squeeze(result, axis=axis)


def _vector_norm(x, ord, axis, keepdims):
    if ord is None or ord == 2:
        return _scaled_norm(x, axis, keepdims)
    if ord == np.inf:
        return np.max(np.abs(x), axis=axis, keepdims=keepdims)
    if ord == -np.inf:
        return np.min(np.abs(x), axis=axis, keepdims=keepdims)
    if ord == 0:
        return np.sum(x != 0, axis=axis, keepdims=keepdims).astype(np.float64)
    if ord == 1:
        return np.sum(np.abs(x), axis=axis, keepdims=keepdims)
    return np.sum(np.abs(x) ** ord, axis=axis, keepdims=keepdims) ** (1.0 / ord)


def _matrix_norm(x, ord, keepdims):
    if ord is None or ord == "fro":
        result = _scaled_norm(x, axis=(-2, -1), keepdims=False)
    elif ord == "nuc":
        result = np.sum(svdvals(x), axis=-1)
    elif ord == 1:
        result = np.max(np.sum(np.abs(x), axis=-2), axis=-1)
    elif ord == -1:
        result = np.min(np.sum(np.abs(x), axis=-2), axis=-1)
    elif ord == np.inf:
        result = np.max(np.sum(np.abs(x), axis=-1), axis=-1)
    elif ord == -np.inf:
        result = np.min(np.sum(np.abs(x), axis=-1), axis=-1)
    elif ord == 2:
        result = np.amax(svdvals(x), axis=-1)
    elif ord == -2:
        result = np.amin(svdvals(x), axis=-1)
    else:
        raise ValueError("Invalid norm order for matrices.")
    if keepdims:
        result = result[..., np.newaxis, np.newaxis]
    return result


def norm(x, ord=None, axis=None, keepdims=False):
    """A vector or matrix norm of `x`.

    With `axis` omitted, a 1-D `x` gets a vector norm and a 2-D `x` gets a matrix norm; with a
    single integer `axis`, a vector norm is taken along that axis (batching over the rest); with
    a 2-tuple `axis`, a matrix norm is taken over those two axes (batching over the rest). `ord`
    selects the kind of norm: for vectors, ``None``/``2`` (Euclidean), ``1``, ``0`` (count of
    nonzeros), `inf`/``-inf``, or any other real `p` (an Lp norm); for matrices, ``None``/``"fro"``
    (Frobenius), ``"nuc"`` (nuclear, the singular values' sum), ``1``/``-1``/`inf`/``-inf`` (the
    largest or smallest absolute column or row sum), or ``2``/``-2`` (the largest or smallest
    singular value).
    """
    x = np.asarray(x)
    single = x.dtype == np.float32 or x.dtype == np.complex64
    real_dtype = np.float32 if single else np.float64
    if x.dtype == np.complex64:
        xf = x.astype(np.complex64)
    elif np.iscomplexobj(x):
        xf = x.astype(np.complex128)
    else:
        xf = x.astype(np.float64)
    if axis is None:
        if xf.ndim == 1:
            result = _vector_norm(xf, ord, None, keepdims)
        elif xf.ndim == 2:
            result = _matrix_norm(xf, ord, keepdims)
        else:
            raise ValueError(
                "Improper number of dimensions to norm."
            )
    elif isinstance(axis, tuple):
        if len(axis) != 2:
            raise ValueError("Invalid norm order for matrices.")
        row_axis, col_axis = (a % xf.ndim for a in axis)
        moved = np.moveaxis(xf, (row_axis, col_axis), (-2, -1))
        result = _matrix_norm(moved, ord, keepdims)
    else:
        result = _vector_norm(xf, ord, axis, keepdims)
    return result.astype(real_dtype)
