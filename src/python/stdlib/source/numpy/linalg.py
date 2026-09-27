"""``numpy.linalg``, ported from NumPy 2.5's ``numpy/linalg/_linalg.py``.

The decompositions and solvers call the native module ``_numpy_linalg``, which stands in for
NumPy's ``_umath_linalg`` gufuncs and raises ``LinAlgError`` itself instead of signalling
through ``errstate(call=...)``. Its ``qr_r_raw`` returns the factored matrix rather than
overwriting its argument. Arrays are never subclasses here, so results are not re-wrapped.

Complex input is not supported yet, so ``eig`` and ``eigvals``, whose results are complex even
for real input, are not provided.
"""

import operator

import _numpy_linalg as _umath_linalg
from _numpy import _LinAlgError as LinAlgError
from _numpy_shape import _normalize_axis_index as normalize_axis_index
from numpy import (
    abs,
    add,
    amax,
    amin,
    argsort,
    array,
    asanyarray,
    asarray,
    atleast_2d,
    cdouble,
    complexfloating,
    count_nonzero,
    csingle,
    divide,
    dot,
    double,
    empty,
    empty_like,
    errstate,
    eye,
    finfo,
    inexact,
    inf,
    intp,
    isnan,
    moveaxis,
    multiply,
    newaxis,
    prod,
    reciprocal,
    single,
    sort,
    sqrt,
    sum,
    swapaxes,
    triu,
    zeros,
)
from numpy import cross as _core_cross
from numpy import diagonal as _core_diagonal
from numpy import matmul as _core_matmul
from numpy import outer as _core_outer
from numpy import tensordot as _core_tensordot
from numpy import trace as _core_trace
from numpy._arraysetops import _Result
from numpy._shape_base import normalize_axis_tuple

__all__ = [
    "matrix_power",
    "solve",
    "tensorsolve",
    "tensorinv",
    "inv",
    "cholesky",
    "eigvalsh",
    "pinv",
    "slogdet",
    "det",
    "svd",
    "svdvals",
    "eigh",
    "lstsq",
    "norm",
    "qr",
    "cond",
    "matrix_rank",
    "LinAlgError",
    "multi_dot",
    "trace",
    "diagonal",
    "cross",
    "outer",
    "tensordot",
    "matmul",
    "matrix_transpose",
    "matrix_norm",
    "vector_norm",
    "vecdot",
]


class EighResult(_Result):
    _fields = ("eigenvalues", "eigenvectors")


class QRResult(_Result):
    _fields = ("Q", "R")


class SlogdetResult(_Result):
    _fields = ("sign", "logabsdet")


class SVDResult(_Result):
    _fields = ("U", "S", "Vh")


class _NoValueType:
    def __repr__(self):
        return "<no value>"


_NoValue = _NoValueType()


def isComplexType(t):
    return issubclass(t, complexfloating)


_real_types_map = {single: single, double: double, csingle: single, cdouble: double}
_complex_types_map = {single: csingle, double: cdouble, csingle: csingle, cdouble: cdouble}


def _realType(t, default=double):
    return _real_types_map.get(t, default)


def _complexType(t, default=cdouble):
    return _complex_types_map.get(t, default)


def _commonType(*arrays):
    result_type = single
    is_complex = False
    for a in arrays:
        type_ = a.dtype.type
        if issubclass(type_, inexact):
            if isComplexType(type_):
                is_complex = True
            rt = _realType(type_, default=None)
            if rt is double:
                result_type = double
            elif rt is None:
                raise TypeError(f"array type {a.dtype.name} is unsupported in linalg")
        else:
            result_type = double
    if is_complex:
        result_type = _complex_types_map[result_type]
        return (cdouble, result_type)
    return (double, result_type)


def _assert_2d(*arrays):
    for a in arrays:
        if a.ndim != 2:
            raise LinAlgError(
                f"{a.ndim}-dimensional array given. Array must be two-dimensional"
            )


def _assert_stacked_2d(*arrays):
    for a in arrays:
        if a.ndim < 2:
            raise LinAlgError(
                f"{a.ndim}-dimensional array given. Array must be at least two-dimensional"
            )


def _assert_stacked_square(*arrays):
    for a in arrays:
        try:
            m, n = a.shape[-2:]
        except ValueError:
            raise LinAlgError(
                f"{a.ndim}-dimensional array given. Array must be at least two-dimensional"
            )
        if m != n:
            raise LinAlgError("Last 2 dimensions of the array must be square")


def _is_empty_2d(arr):
    return arr.size == 0 and prod(arr.shape[-2:]) == 0


def transpose(a):
    return swapaxes(a, -1, -2)


def tensorsolve(a, b, axes=None):
    a = asarray(a)
    b = asarray(b)
    an = a.ndim
    if axes is not None:
        allaxes = list(range(an))
        for k in axes:
            allaxes.remove(k)
            allaxes.insert(an, k)
        a = a.transpose(allaxes)
    oldshape = a.shape[-(an - b.ndim) :]
    prod = 1
    for k in oldshape:
        prod *= k
    if a.size != prod**2:
        raise LinAlgError(
            "Input arrays must satisfy the requirement "
            "            prod(a.shape[b.ndim:]) == prod(a.shape[:b.ndim])"
        )
    a = a.reshape(prod, prod)
    b = b.ravel()
    res = solve(a, b)
    return res.reshape(oldshape)


def solve(a, b):
    a = asarray(a)
    _assert_stacked_square(a)
    b = asarray(b)
    t, result_t = _commonType(a, b)
    if b.ndim == 1:
        gufunc = _umath_linalg.solve1
    else:
        gufunc = _umath_linalg.solve
    r = gufunc(a, b)
    return r.astype(result_t, copy=False)


def tensorinv(a, ind=2):
    a = asarray(a)
    oldshape = a.shape
    prod = 1
    if ind > 0:
        invshape = oldshape[ind:] + oldshape[:ind]
        for k in oldshape[ind:]:
            prod *= k
    else:
        raise ValueError("Invalid ind argument.")
    a = a.reshape(prod, -1)
    ia = inv(a)
    return ia.reshape(*invshape)


def inv(a):
    a = asarray(a)
    _assert_stacked_square(a)
    t, result_t = _commonType(a)
    ainv = _umath_linalg.inv(a)
    return ainv.astype(result_t, copy=False)


def matrix_power(a, n):
    a = asanyarray(a)
    _assert_stacked_square(a)
    try:
        n = operator.index(n)
    except TypeError as e:
        raise TypeError("exponent must be an integer") from e
    if a.dtype != object:
        fmatmul = matmul
    elif a.ndim == 2:
        fmatmul = dot
    else:
        raise NotImplementedError("matrix_power not supported for stacks of object arrays")
    if n == 0:
        a = empty_like(a)
        a[...] = eye(a.shape[-2], dtype=a.dtype)
        return a
    elif n < 0:
        a = inv(a)
        n = abs(n)
    if n == 1:
        return a
    elif n == 2:
        return fmatmul(a, a)
    elif n == 3:
        return fmatmul(fmatmul(a, a), a)
    z = result = None
    while n > 0:
        z = a if z is None else fmatmul(z, z)
        n, bit = divmod(n, 2)
        if bit:
            result = z if result is None else fmatmul(result, z)
    return result


def cholesky(a, /, *, upper=False):
    gufunc = _umath_linalg.cholesky_up if upper else _umath_linalg.cholesky_lo
    a = asarray(a)
    _assert_stacked_square(a)
    t, result_t = _commonType(a)
    r = gufunc(a)
    return r.astype(result_t, copy=False)


def outer(x1, x2, /):
    x1 = asanyarray(x1)
    x2 = asanyarray(x2)
    if x1.ndim != 1 or x2.ndim != 1:
        raise ValueError(
            "Input arrays must be one-dimensional, but they are "
            f"x1.ndim={x1.ndim!r} and x2.ndim={x2.ndim!r}."
        )
    return _core_outer(x1, x2, out=None)


def qr(a, mode="reduced"):
    if mode not in ("reduced", "complete", "r", "raw"):
        if mode in ("f", "full"):
            import warnings

            msg = (
                "The 'full' option is deprecated in favor of 'reduced'.\n"
                "For backward compatibility let mode default."
            )
            warnings.warn(msg, DeprecationWarning, stacklevel=2)
            mode = "reduced"
        elif mode in ("e", "economic"):
            import warnings

            msg = "The 'economic' option is deprecated."
            warnings.warn(msg, DeprecationWarning, stacklevel=2)
            mode = "economic"
        else:
            raise ValueError(f"Unrecognized mode '{mode}'")
    a = asarray(a)
    _assert_stacked_2d(a)
    m, n = a.shape[-2:]
    t, result_t = _commonType(a)
    mn = min(m, n)
    a, tau = _umath_linalg.qr_r_raw(a)
    if mode == "r":
        r = triu(a[..., :mn, :])
        r = r.astype(result_t, copy=False)
        return r
    if mode == "raw":
        q = transpose(a)
        q = q.astype(result_t, copy=False)
        tau = tau.astype(result_t, copy=False)
        return (q, tau)
    if mode == "economic":
        a = a.astype(result_t, copy=False)
        return a
    if mode == "complete" and m > n:
        mc = m
        gufunc = _umath_linalg.qr_complete
    else:
        mc = mn
        gufunc = _umath_linalg.qr_reduced
    q = gufunc(a, tau)
    r = triu(a[..., :mc, :])
    q = q.astype(result_t, copy=False)
    r = r.astype(result_t, copy=False)
    return QRResult(q, r)


def eigvalsh(a, UPLO="L"):
    UPLO = UPLO.upper()
    if UPLO not in ("L", "U"):
        raise ValueError("UPLO argument must be 'L' or 'U'")
    if UPLO == "L":
        gufunc = _umath_linalg.eigvalsh_lo
    else:
        gufunc = _umath_linalg.eigvalsh_up
    a = asarray(a)
    _assert_stacked_square(a)
    t, result_t = _commonType(a)
    w = gufunc(a)
    return w.astype(_realType(result_t), copy=False)


def eigh(a, UPLO="L"):
    UPLO = UPLO.upper()
    if UPLO not in ("L", "U"):
        raise ValueError("UPLO argument must be 'L' or 'U'")
    a = asarray(a)
    _assert_stacked_square(a)
    t, result_t = _commonType(a)
    if UPLO == "L":
        gufunc = _umath_linalg.eigh_lo
    else:
        gufunc = _umath_linalg.eigh_up
    w, vt = gufunc(a)
    w = w.astype(_realType(result_t), copy=False)
    vt = vt.astype(result_t, copy=False)
    return EighResult(w, vt)


def svd(a, full_matrices=True, compute_uv=True, hermitian=False):
    import numpy as np

    a = asarray(a)
    if hermitian:
        if compute_uv:
            s, u = eigh(a)
            sgn = np.copysign(1.0, s)
            s = abs(s)
            sidx = argsort(s)[..., ::-1]
            sgn = np.take_along_axis(sgn, sidx, axis=-1)
            s = np.take_along_axis(s, sidx, axis=-1)
            u = np.take_along_axis(u, sidx[..., None, :], axis=-1)
            vt = transpose(u * sgn[..., None, :]).conjugate()
            return SVDResult(u, s, vt)
        else:
            s = eigvalsh(a)
            s = abs(s)
            return sort(s)[..., ::-1]
    _assert_stacked_2d(a)
    t, result_t = _commonType(a)
    if compute_uv:
        if full_matrices:
            gufunc = _umath_linalg.svd_f
        else:
            gufunc = _umath_linalg.svd_s
        u, s, vh = gufunc(a)
        u = u.astype(result_t, copy=False)
        s = s.astype(_realType(result_t), copy=False)
        vh = vh.astype(result_t, copy=False)
        return SVDResult(u, s, vh)
    s = _umath_linalg.svd(a)
    s = s.astype(_realType(result_t), copy=False)
    return s


def svdvals(x, /):
    return svd(x, compute_uv=False, hermitian=False)


def cond(x, p=None):
    x = asarray(x)
    if _is_empty_2d(x):
        raise LinAlgError("cond is not defined on empty arrays")
    if p is None or p in {2, -2}:
        s = svd(x, compute_uv=False)
        with errstate(all="ignore"):
            if p == -2:
                r = s[..., -1] / s[..., 0]
            else:
                r = s[..., 0] / s[..., -1]
    else:
        _assert_stacked_square(x)
        t, result_t = _commonType(x)
        result_t = _realType(result_t)
        with errstate(all="ignore"):
            invx = _umath_linalg.inv(x)
            r = norm(x, p, axis=(-2, -1)) * norm(invx, p, axis=(-2, -1))
        r = r.astype(result_t, copy=False)
    nan_mask = isnan(r)
    if nan_mask.any():
        nan_mask &= ~isnan(x).any(axis=(-2, -1))
        if r.ndim > 0:
            r[nan_mask] = inf
        elif nan_mask:
            r = r.dtype.type(inf)
    return r


def matrix_rank(A, tol=None, hermitian=False, *, rtol=None):
    if rtol is not None and tol is not None:
        raise ValueError("`tol` and `rtol` can't be both set.")
    A = asarray(A)
    if A.ndim < 2:
        return int(not all(A == 0))
    S = svd(A, compute_uv=False, hermitian=hermitian)
    if tol is None:
        if rtol is None:
            rtol = max(A.shape[-2:]) * finfo(S.dtype).eps
        else:
            rtol = asarray(rtol)[..., newaxis]
        tol = S.max(axis=-1, keepdims=True, initial=0) * rtol
    else:
        tol = asarray(tol)[..., newaxis]
    return count_nonzero(S > tol, axis=-1)


def pinv(a, rcond=None, hermitian=False, *, rtol=_NoValue):
    a = asarray(a)
    if rcond is None:
        if rtol is _NoValue:
            rcond = 1e-15
        elif rtol is None:
            rcond = max(a.shape[-2:]) * finfo(a.dtype).eps
        else:
            rcond = rtol
    elif rtol is not _NoValue:
        raise ValueError("`rtol` and `rcond` can't be both set.")
    rcond = asarray(rcond)
    if _is_empty_2d(a):
        m, n = a.shape[-2:]
        res = empty(a.shape[:-2] + (n, m), dtype=a.dtype)
        return res
    a = a.conjugate()
    u, s, vt = svd(a, full_matrices=False, hermitian=hermitian)
    cutoff = rcond[..., newaxis] * amax(s, axis=-1, keepdims=True)
    large = s > cutoff
    s = divide(1, s, where=large, out=s)
    s[~large] = 0
    res = matmul(transpose(vt), multiply(s[..., newaxis], transpose(u)))
    return res


def slogdet(a):
    a = asarray(a)
    _assert_stacked_square(a)
    t, result_t = _commonType(a)
    real_t = _realType(result_t)
    sign, logdet = _umath_linalg.slogdet(a)
    sign = sign.astype(result_t, copy=False)
    logdet = logdet.astype(real_t, copy=False)
    return SlogdetResult(sign, logdet)


def det(a):
    a = asarray(a)
    _assert_stacked_square(a)
    t, result_t = _commonType(a)
    r = _umath_linalg.det(a)
    r = r.astype(result_t, copy=False)
    return r


def lstsq(a, b, rcond=None):
    a = asarray(a)
    b = asarray(b)
    is_1d = b.ndim == 1
    if is_1d:
        b = b[:, newaxis]
    _assert_2d(a, b)
    m, n = a.shape[-2:]
    m2, n_rhs = b.shape[-2:]
    if m != m2:
        raise LinAlgError("Incompatible dimensions")
    t, result_t = _commonType(a, b)
    result_real_t = _realType(result_t)
    if rcond is None:
        rcond = finfo(t).eps * max(n, m)
    if n_rhs == 0:
        b = zeros(b.shape[:-2] + (m, n_rhs + 1), dtype=b.dtype)
    x, resids, rank, s = _umath_linalg.lstsq(a, b, rcond)
    if m == 0:
        x[...] = 0
    if n_rhs == 0:
        x = x[..., :n_rhs]
        resids = resids[..., :n_rhs]
    if is_1d:
        x = x.squeeze(axis=-1)
    if rank != n or m <= n:
        resids = array([], result_real_t)
    s = s.astype(result_real_t, copy=False)
    resids = resids.astype(result_real_t, copy=False)
    x = x.astype(result_t, copy=True)
    return (x, resids, rank, s)


def _multi_svd_norm(x, row_axis, col_axis, op, initial=None):
    y = moveaxis(x, (row_axis, col_axis), (-2, -1))
    result = op(svd(y, compute_uv=False), axis=-1, initial=initial)
    return result


def norm(x, ord=None, axis=None, keepdims=False):
    x = asarray(x)
    # NumPy tests `issubclass(x.dtype.type, (inexact, object_))`. Here `object_` is the
    # builtin `object`, of which every scalar type is a subclass, so test the kind instead.
    if x.dtype.kind not in "fcO":
        x = x.astype(float)
    if axis is None:
        ndim = x.ndim
        if (
            ord is None
            or (ord in ("f", "fro") and ndim == 2)
            or (ord == 2 and ndim == 1)
        ):
            x = x.ravel(order="K")
            if isComplexType(x.dtype.type):
                x_real = x.real
                x_imag = x.imag
                sqnorm = x_real.dot(x_real) + x_imag.dot(x_imag)
            else:
                sqnorm = x.dot(x)
            ret = sqrt(sqnorm)
            if keepdims:
                ret = ret.reshape(ndim * [1])
            return ret
    nd = x.ndim
    if axis is None:
        axis = tuple(range(nd))
    elif not isinstance(axis, tuple):
        try:
            axis = int(axis)
        except Exception as e:
            raise TypeError("'axis' must be None, an integer or a tuple of integers") from e
        axis = (axis,)
    if len(axis) == 1:
        if ord == inf:
            return abs(x).max(axis=axis, keepdims=keepdims, initial=0)
        elif ord == -inf:
            return abs(x).min(axis=axis, keepdims=keepdims)
        elif ord == 0:
            return (x != 0).astype(x.real.dtype).sum(axis=axis, keepdims=keepdims)
        elif ord == 1:
            return add.reduce(abs(x), axis=axis, keepdims=keepdims)
        elif ord is None or ord == 2:
            s = (x.conj() * x).real
            return sqrt(add.reduce(s, axis=axis, keepdims=keepdims))
        elif isinstance(ord, str):
            raise ValueError(f"Invalid norm order '{ord}' for vectors")
        else:
            absx = abs(x)
            absx **= ord
            ret = add.reduce(absx, axis=axis, keepdims=keepdims)
            ret **= reciprocal(ord, dtype=ret.dtype)
            return ret
    elif len(axis) == 2:
        row_axis, col_axis = axis
        row_axis = normalize_axis_index(row_axis, nd)
        col_axis = normalize_axis_index(col_axis, nd)
        if row_axis == col_axis:
            raise ValueError("Duplicate axes given.")
        if ord == 2:
            ret = _multi_svd_norm(x, row_axis, col_axis, amax, 0)
        elif ord == -2:
            ret = _multi_svd_norm(x, row_axis, col_axis, amin)
        elif ord == 1:
            if col_axis > row_axis:
                col_axis -= 1
            ret = add.reduce(abs(x), axis=row_axis).max(axis=col_axis, initial=0)
        elif ord == inf:
            if row_axis > col_axis:
                row_axis -= 1
            ret = add.reduce(abs(x), axis=col_axis).max(axis=row_axis, initial=0)
        elif ord == -1:
            if col_axis > row_axis:
                col_axis -= 1
            ret = add.reduce(abs(x), axis=row_axis).min(axis=col_axis)
        elif ord == -inf:
            if row_axis > col_axis:
                row_axis -= 1
            ret = add.reduce(abs(x), axis=col_axis).min(axis=row_axis)
        elif ord in [None, "fro", "f"]:
            ret = sqrt(add.reduce((x.conj() * x).real, axis=axis))
        elif ord == "nuc":
            ret = _multi_svd_norm(x, row_axis, col_axis, sum, 0)
        else:
            raise ValueError("Invalid norm order for matrices.")
        if keepdims:
            ret_shape = list(x.shape)
            ret_shape[axis[0]] = 1
            ret_shape[axis[1]] = 1
            ret = ret.reshape(ret_shape)
        return ret
    else:
        raise ValueError("Improper number of dimensions to norm.")


def multi_dot(arrays, *, out=None):
    n = len(arrays)
    if n < 2:
        raise ValueError("Expecting at least two arrays.")
    elif n == 2:
        return dot(arrays[0], arrays[1], out=out)
    arrays = [asanyarray(a) for a in arrays]
    ndim_first, ndim_last = (arrays[0].ndim, arrays[-1].ndim)
    if arrays[0].ndim == 1:
        arrays[0] = atleast_2d(arrays[0])
    if arrays[-1].ndim == 1:
        arrays[-1] = atleast_2d(arrays[-1]).T
    _assert_2d(*arrays)
    if n == 3:
        result = _multi_dot_three(arrays[0], arrays[1], arrays[2], out=out)
    else:
        order = _multi_dot_matrix_chain_order(arrays)
        result = _multi_dot(arrays, order, 0, n - 1, out=out)
    if ndim_first == 1 and ndim_last == 1:
        return result[0, 0]
    elif ndim_first == 1 or ndim_last == 1:
        return result.ravel()
    return result


def _multi_dot_three(A, B, C, out=None):
    a0, a1b0 = A.shape
    b1c0, c1 = C.shape
    cost1 = a0 * b1c0 * (a1b0 + c1)
    cost2 = a1b0 * c1 * (a0 + b1c0)
    if cost1 < cost2:
        return dot(dot(A, B), C, out=out)
    return dot(A, dot(B, C), out=out)


def _multi_dot_matrix_chain_order(arrays, return_costs=False):
    n = len(arrays)
    p = [a.shape[0] for a in arrays] + [arrays[-1].shape[1]]
    m = zeros((n, n), dtype=double)
    s = empty((n, n), dtype=intp)
    for l in range(1, n):
        for i in range(n - l):
            j = i + l
            m[i, j] = inf
            for k in range(i, j):
                q = m[i, k] + m[k + 1, j] + p[i] * p[k + 1] * p[j + 1]
                if q < m[i, j]:
                    m[i, j] = q
                    s[i, j] = k
    return (s, m) if return_costs else s


def _multi_dot(arrays, order, i, j, out=None):
    if i == j:
        assert out is None
        return arrays[i]
    return dot(
        _multi_dot(arrays, order, i, order[i, j]),
        _multi_dot(arrays, order, order[i, j] + 1, j),
        out=out,
    )


def diagonal(x, /, *, offset=0):
    return _core_diagonal(x, offset, axis1=-2, axis2=-1)


def trace(x, /, *, offset=0, dtype=None):
    return _core_trace(x, offset, axis1=-2, axis2=-1, dtype=dtype)


def cross(x1, x2, /, *, axis=-1):
    x1 = asanyarray(x1)
    x2 = asanyarray(x2)
    return _core_cross(x1, x2, axis=axis)


def matmul(x1, x2, /):
    return _core_matmul(x1, x2)


def tensordot(x1, x2, /, *, axes=2):
    return _core_tensordot(x1, x2, axes=axes)


def matrix_transpose(x, /):
    x = asanyarray(x)
    if x.ndim < 2:
        raise ValueError(f"Input array must be at least 2-dimensional, but it is {x.ndim}")
    return swapaxes(x, -1, -2)


def matrix_norm(x, /, *, keepdims=False, ord="fro"):
    x = asanyarray(x)
    return norm(x, axis=(-2, -1), keepdims=keepdims, ord=ord)


def vector_norm(x, /, *, axis=None, keepdims=False, ord=2):
    x = asanyarray(x)
    shape = list(x.shape)
    if axis is None:
        x = x.ravel()
        _axis = 0
    elif isinstance(axis, tuple):
        normalized_axis = normalize_axis_tuple(axis, x.ndim)
        rest = tuple(i for i in range(x.ndim) if i not in normalized_axis)
        newshape = axis + rest
        x = x.transpose(newshape).reshape(
            (prod([x.shape[i] for i in axis], dtype=int), *[x.shape[i] for i in rest])
        )
        _axis = 0
    else:
        _axis = axis
    res = norm(x, axis=_axis, ord=ord)
    if keepdims:
        _axis = normalize_axis_tuple(range(len(shape)) if axis is None else axis, len(shape))
        for i in _axis:
            shape[i] = 1
        res = res.reshape(tuple(shape))
    return res


def vecdot(x1, x2, /, *, axis=-1):
    """The ``(n),(n)->()`` gufunc ``numpy.vecdot``: sums of ``conj(x1) * x2`` along ``axis``."""
    x1 = asanyarray(x1)
    x2 = asanyarray(x2)
    for operand, x in enumerate((x1, x2)):
        if x.ndim == 0:
            raise ValueError(
                f"vecdot: Input operand {operand} does not have enough dimensions (has 0, "
                "gufunc core with signature (n),(n)->() requires 1)"
            )
    x1 = moveaxis(x1, axis, -1)
    x2 = moveaxis(x2, axis, -1)
    if x1.shape[-1] != x2.shape[-1]:
        raise ValueError(
            "vecdot: Input operand 1 has a mismatch in its core dimension 0, with gufunc "
            f"signature (n),(n)->() (size {x2.shape[-1]} is different from {x1.shape[-1]})"
        )
    return (x1.conj() * x2).sum(axis=-1)
