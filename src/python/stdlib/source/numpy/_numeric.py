"""General numeric helpers built on the native array, ufunc and reduction primitives: triangles,
counting, closeness, dtype classification, and the small ``ndarray``-method wrappers
(``round``/``clip``) exposed as top-level functions.
"""

import numpy as np

__all__ = [
    "allclose",
    "argwhere",
    "around",
    "array_equal",
    "array_equiv",
    "asarray_chkfinite",
    "astype",
    "clip",
    "common_type",
    "concat",
    "count_nonzero",
    "cumulative_prod",
    "cumulative_sum",
    "flatnonzero",
    "imag",
    "indices",
    "iscomplexobj",
    "isclose",
    "isdtype",
    "isfortran",
    "isneginf",
    "isposinf",
    "isrealobj",
    "iterable",
    "mintypecode",
    "nan_to_num",
    "ptp",
    "real",
    "real_if_close",
    "round",
    "tri",
    "tril",
    "triu",
]

concat = np.concatenate


def tri(N, M=None, k=0, dtype=np.float64):
    """An N-by-M array that is 1 at and below diagonal `k`, 0 elsewhere."""
    M = N if M is None else M
    rows = np.arange(N).reshape(N, 1)
    cols = np.arange(M).reshape(1, M)
    return (cols <= rows + k).astype(dtype)


def tril(m, k=0):
    """`m` with the elements above diagonal `k` zeroed."""
    m = np.asanyarray(m)
    mask = tri(m.shape[-2], m.shape[-1], k=k, dtype=np.bool_)
    return np.where(mask, m, 0)


def triu(m, k=0):
    """`m` with the elements below diagonal `k` zeroed."""
    m = np.asanyarray(m)
    mask = tri(m.shape[-2], m.shape[-1], k=k - 1, dtype=np.bool_)
    return np.where(~mask, m, 0)


def indices(dimensions, dtype=np.int64, sparse=False):
    """The per-axis index arrays of an array with shape `dimensions`."""
    dimensions = tuple(dimensions)
    ndim = len(dimensions)
    axes = [
        np.arange(size, dtype=dtype).reshape([size if i == axis else 1 for i in range(ndim)])
        for axis, size in enumerate(dimensions)
    ]
    if sparse:
        return tuple(axes)
    return np.stack([np.broadcast_to(a, dimensions) for a in axes], axis=0)


def count_nonzero(a, axis=None, *, keepdims=False):
    """The number of elements of `a` that are not the dtype's zero value."""
    a = np.asanyarray(a)
    return np.sum(a != 0, axis=axis, keepdims=keepdims, dtype=np.int64)


def flatnonzero(a):
    """The indices of the nonzero elements of `a`, as if it were raveled."""
    return np.nonzero(np.ravel(a))[0]


def argwhere(a):
    """One row per nonzero element of `a`, giving its coordinates."""
    a = np.asanyarray(a)
    if a.ndim == 0:
        a = a.reshape(1)
    return np.asanyarray(np.nonzero(a)).T


def ptp(a, axis=None, out=None, keepdims=False):
    """The range (max - min) of `a`, along `axis` if given."""
    result = np.max(a, axis=axis, keepdims=keepdims) - np.min(a, axis=axis, keepdims=keepdims)
    if out is not None:
        out[...] = result
        return out
    return result


def round(a, decimals=0, out=None):
    """`a` rounded to `decimals` places, ties to even, at `a`'s own dtype and precision."""
    return np.asanyarray(a).round(decimals=decimals, out=out)


around = round


def clip(a, a_min=None, a_max=None, out=None, *, min=None, max=None, **kwargs):
    """`a` with every value clamped to ``[lower, upper]``; either bound may be ``None``."""
    lower = min if min is not None else a_min
    upper = max if max is not None else a_max
    return np.asanyarray(a).clip(min=lower, max=upper, out=out, **kwargs)


def isclose(a, b, rtol=1e-05, atol=1e-08, equal_nan=False):
    """Elementwise ``|a - b| <= atol + rtol * |b|``, with matching infinities counted as close."""
    a_b, b_b = np.broadcast_arrays(np.asanyarray(a), np.asanyarray(b))
    with np.errstate(invalid="ignore"):
        close = np.abs(a_b - b_b) <= (atol + rtol * np.abs(b_b))
    inf_mask = np.isinf(a_b) | np.isinf(b_b)
    if np.any(inf_mask):
        same_sign_inf = np.isinf(a_b) & np.isinf(b_b) & (np.signbit(a_b) == np.signbit(b_b))
        close = np.where(inf_mask, same_sign_inf, close)
    if equal_nan:
        close = np.where(np.isnan(a_b) & np.isnan(b_b), True, close)
    return close


def allclose(a, b, rtol=1e-05, atol=1e-08, equal_nan=False):
    """Whether every element of `a` and `b` is close under :func:`isclose`."""
    return bool(np.all(isclose(a, b, rtol=rtol, atol=atol, equal_nan=equal_nan)))


def array_equal(a1, a2, equal_nan=False):
    """Whether `a1` and `a2` have the same shape and, elementwise, the same values."""
    a1 = np.asanyarray(a1)
    a2 = np.asanyarray(a2)
    if a1.shape != a2.shape:
        return False
    if equal_nan and a1.dtype.kind in "fc" and a2.dtype.kind in "fc":
        same = (a1 == a2) | (np.isnan(a1) & np.isnan(a2))
        return bool(np.all(same))
    return bool(np.all(a1 == a2))


def array_equiv(a1, a2):
    """Whether `a1` and `a2` are equal once broadcast to a common shape."""
    try:
        b1, b2 = np.broadcast_arrays(np.asanyarray(a1), np.asanyarray(a2))
    except ValueError:
        return False
    return bool(np.all(b1 == b2))


def iterable(obj):
    """Whether ``iter(obj)`` succeeds."""
    try:
        iter(obj)
    except TypeError:
        return False
    return True


def isfortran(a):
    """Whether `a` is stored in Fortran (column-major) order and not also C order."""
    flags = np.asanyarray(a).flags
    return bool(flags.f_contiguous and not flags.c_contiguous)


def asarray_chkfinite(a, dtype=None, order=None):
    """``asarray(a)``, rejecting an array that contains a NaN or an infinity."""
    result = np.asarray(a, dtype=dtype, order=order)
    if not bool(np.all(np.isfinite(result))):
        raise ValueError("array must not contain infs or NaNs")
    return result


def iscomplexobj(x):
    """Whether `x` (a value or array-like) has a complex dtype."""
    return np.asanyarray(x).dtype.kind == "c"


def isrealobj(x):
    """The negation of :func:`iscomplexobj`."""
    return not iscomplexobj(x)


def real(val):
    """The real part of `val`; a NumPy array or scalar keeps its own type."""
    if isinstance(val, (np.ndarray, np.generic)):
        return val.real
    return np.asanyarray(val).real


def imag(val):
    """The imaginary part of `val` (zero, at `val`'s dtype, for a real value)."""
    if isinstance(val, (np.ndarray, np.generic)):
        return val.imag
    return np.asanyarray(val).imag


def isposinf(x):
    """Whether each element of `x` is positive infinity."""
    x = np.asanyarray(x)
    return np.isinf(x) & (x > 0)


def isneginf(x):
    """Whether each element of `x` is negative infinity."""
    x = np.asanyarray(x)
    return np.isinf(x) & (x < 0)


def real_if_close(a, tol=100):
    """The real part of `a` if every imaginary part is within `tol` epsilons of zero."""
    a = np.asanyarray(a)
    if a.dtype.kind != "c":
        return a
    limit = tol * np.finfo(a.dtype).eps
    if bool(np.all(np.abs(a.imag) < limit)):
        return a.real
    return a


def _fill_nonfinite(values, nan, posinf, neginf):
    info = np.finfo(values.dtype)
    pos = info.max if posinf is None else posinf
    neg = info.min if neginf is None else neginf
    values[np.isnan(values)] = nan
    values[np.isposinf(values)] = pos
    values[np.isneginf(values)] = neg


def nan_to_num(x, copy=True, nan=0.0, posinf=None, neginf=None):
    """`x` with NaN, +inf and -inf replaced by finite values."""
    result = np.array(x, copy=True) if copy else np.asanyarray(x)
    if result.dtype.kind == "c":
        _fill_nonfinite(result.real, nan, posinf, neginf)
        _fill_nonfinite(result.imag, nan, posinf, neginf)
    elif result.dtype.kind == "f":
        _fill_nonfinite(result, nan, posinf, neginf)
    return result


def astype(x, dtype, /, *, copy=True, device=None):
    """``x.astype(dtype)`` for an array or NumPy scalar; rejects other Python values."""
    if not isinstance(x, (np.ndarray, np.generic)):
        raise TypeError(f"Input should be a NumPy array or scalar. It is a {type(x)} instead.")
    return x.astype(dtype, copy=copy)


_ISDTYPE_KINDS = {
    "bool": lambda dtype: dtype.kind == "b",
    "signed integer": lambda dtype: dtype.kind == "i",
    "unsigned integer": lambda dtype: dtype.kind == "u",
    "integral": lambda dtype: dtype.kind in "iu",
    "real floating": lambda dtype: dtype.kind == "f",
    "complex floating": lambda dtype: dtype.kind == "c",
    "numeric": lambda dtype: dtype.kind in "iufc",
}


def _is_dtype_class(value):
    return isinstance(value, type) and issubclass(value, np.generic)


def isdtype(dtype, kind):
    """Whether `dtype` belongs to `kind`, an Array API kind name, dtype, or tuple of either.

    Abstract scalar types (``np.floating``, ``np.integer``, ``np.number``, ...) are accepted as
    `dtype` but resolve to no concrete dtype, so they belong to no kind.
    """
    if not isinstance(dtype, np.dtype) and not _is_dtype_class(dtype):
        raise TypeError("dtype argument must be a NumPy dtype")
    if isinstance(dtype, np.dtype):
        resolved = dtype
    else:
        try:
            resolved = np.dtype(dtype)
        except TypeError:
            resolved = None
    kinds = kind if isinstance(kind, tuple) else (kind,)
    for one_kind in kinds:
        if isinstance(one_kind, str):
            if one_kind not in _ISDTYPE_KINDS:
                raise ValueError(f"{one_kind!r} is not a known kind name")
            if resolved is not None and _ISDTYPE_KINDS[one_kind](resolved):
                return True
        elif isinstance(one_kind, np.dtype) or _is_dtype_class(one_kind):
            if resolved is not None and resolved == np.dtype(one_kind):
                return True
        else:
            raise TypeError("kind argument must be comprised of NumPy dtypes")
    return False


# `mintypecode`'s default typeset, "GDFgdf", ordered from narrowest to widest: real precisions
# (float32 < float64 < long double) and complex precisions (complex64 < complex128 < clong
# double), with the two chains interleaved so `max` picks the widest type present. shellsim has
# no long-double dtype, but 'g'/'G' are still valid *characters* here: `mintypecode` only reasons
# about typecodes, never constructs an array of them.
_MINTYPECODE_ORDER = "fdgFDG"


def mintypecode(typechars, typeset="GDFgdf", default="d"):
    """The character in `typeset` that every dtype in `typechars` can safely cast to.

    Determined empirically against NumPy 2.5.3 (no public spec covers the exact promotion): the
    widest type in `typechars` wins, by `_MINTYPECODE_ORDER`, with one documented irregularity
    NumPy itself carries - a double (``'d'``) alongside a single complex (``'F'``) promotes
    straight to double complex (``'D'``) even when a wider real type (long double, ``'g'``) or
    wider complex type (``'G'``) is also present.
    """
    chars = [t if isinstance(t, str) else np.asanyarray(t).dtype.char for t in typechars]
    intersection = [c for c in chars if c in typeset]
    if not intersection:
        return default
    if "F" in intersection and "d" in intersection:
        return "D"
    return max(intersection, key=_MINTYPECODE_ORDER.index)


def common_type(*arrays):
    """The smallest inexact NumPy scalar type every array in `arrays` can be cast to."""
    is_complex = False
    precision = 16
    for a in arrays:
        dtype = np.asanyarray(a).dtype
        if dtype.kind == "c":
            is_complex = True
            precision = max(precision, dtype.itemsize * 4)
        elif dtype.kind == "f":
            precision = max(precision, dtype.itemsize * 8)
        elif dtype.kind in "iu":
            precision = max(precision, 64)
        else:
            raise TypeError("can't get common type for non-numeric array")
    if is_complex:
        return np.complex64 if precision <= 32 else np.complex128
    if precision <= 16:
        return np.float16
    if precision <= 32:
        return np.float32
    return np.float64


def cumulative_sum(x, axis=None, dtype=None, out=None, include_initial=False):
    """The Array API's ``cumulative_sum``: :func:`numpy.cumsum`, optionally with a leading zero."""
    return _cumulative(np.cumsum, x, axis, dtype, out, include_initial, identity=0)


def cumulative_prod(x, axis=None, dtype=None, out=None, include_initial=False):
    """The Array API's ``cumulative_prod``: :func:`numpy.cumprod`, optionally with a leading one."""
    return _cumulative(np.cumprod, x, axis, dtype, out, include_initial, identity=1)


def _cumulative(func, x, axis, dtype, out, include_initial, identity):
    x = np.asanyarray(x)
    if axis is None:
        if x.ndim > 1:
            raise ValueError("For arrays which have more than one dimension ``axis`` argument is required.")
        axis = 0
    result = func(x, axis=axis, dtype=dtype)
    if include_initial:
        shape = list(result.shape)
        shape[axis] = 1
        pad = np.full(shape, identity, dtype=result.dtype)
        result = np.concatenate([pad, result], axis=axis)
    if out is not None:
        out[...] = result
        return out
    return result
