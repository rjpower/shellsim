"""Whole-array numeric helpers: closeness and equality (``isclose``/``allclose``,
``array_equal``), dtype classification, the small ``ndarray``-method wrappers (``round``/
``clip``) exposed as top-level functions, difference and integration (``diff``, ``gradient``,
``trapezoid``), interpolation and correlation (``interp``, ``convolve``/``correlate``),
polynomial least-squares fitting (``polyfit``/``polyval``), the condition-driven selectors
(``select``, ``extract``, ``place``, ``putmask``), and ``vectorize``.
"""

import re

import numpy as np
from _numpy_math import _compiled_interp, _compiled_interp_complex, _correlate

__all__ = [
    "allclose",
    "angle",
    "argwhere",
    "around",
    "array_equal",
    "array_equiv",
    "asarray_chkfinite",
    "astype",
    "clip",
    "common_type",
    "concat",
    "convolve",
    "correlate",
    "count_nonzero",
    "cross",
    "cumulative_prod",
    "cumulative_sum",
    "diff",
    "extract",
    "flatnonzero",
    "gradient",
    "imag",
    "interp",
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
    "place",
    "polyfit",
    "polyval",
    "ptp",
    "putmask",
    "real",
    "real_if_close",
    "round",
    "select",
    "trapezoid",
    "vecdot",
    "vectorize",
]

concat = np.concatenate


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


def angle(z, deg=False):
    """The phase of `z` (real input is treated as having a zero imaginary part)."""
    z = np.asanyarray(z)
    if z.dtype.kind == "c":
        result = np.arctan2(z.imag, z.real)
    else:
        result = np.arctan2(np.zeros(z.shape, dtype=np.float64), z.astype(np.float64))
    return result * (180.0 / np.pi) if deg else result


def _diff_edge(a, axis, value):
    """`value` broadcast to a length-1 slab of `a` along `axis`, for `diff`'s prepend/append."""
    value = np.asanyarray(value)
    if value.ndim != 0:
        return value
    shape = list(a.shape)
    shape[axis] = 1
    return np.broadcast_to(value, shape)


def diff(a, n=1, axis=-1, prepend=None, append=None):
    """The `n`-th discrete difference of `a` along `axis` (a boolean array uses ``!=``)."""
    a = np.asanyarray(a)
    if n < 0:
        raise ValueError(f"order must be non-negative but got {n}")
    axis = axis % a.ndim if a.ndim else axis
    if prepend is not None:
        a = np.concatenate([_diff_edge(a, axis, prepend), a], axis=axis)
    if append is not None:
        a = np.concatenate([a, _diff_edge(a, axis, append)], axis=axis)
    for _ in range(n):
        upper = [slice(None)] * a.ndim
        lower = [slice(None)] * a.ndim
        upper[axis] = slice(1, None)
        lower[axis] = slice(None, -1)
        upper_part, lower_part = a[tuple(upper)], a[tuple(lower)]
        a = (upper_part != lower_part) if a.dtype == np.bool_ else (upper_part - lower_part)
    return a


def trapezoid(y, x=None, dx=1.0, axis=-1):
    """The integral of `y` along `axis` by the trapezoidal rule, over sample points `x` or a
    uniform spacing `dx`.

    Interval ``i`` contributes ``d[i] * (y[i + 1] + y[i]) / 2.0`` and the contributions are
    summed along `axis`. A 1-D `x` spaces that axis; an `x` of more dimensions is differenced
    along `axis` and broadcasts against `y`, as `dx` does.
    """
    y = np.asanyarray(y)
    if x is None:
        d = dx
    else:
        x = np.asanyarray(x)
        if x.ndim == 1:
            d = np.diff(x)
            shape = [1] * y.ndim
            shape[axis] = d.shape[0]
            d = d.reshape(shape)
        else:
            d = np.diff(x, axis=axis)
    upper = [slice(None)] * y.ndim
    lower = [slice(None)] * y.ndim
    upper[axis] = slice(1, None)
    lower[axis] = slice(None, -1)
    return (d * (y[tuple(upper)] + y[tuple(lower)]) / 2.0).sum(axis)


def gradient(f, *varargs, axis=None, edge_order=1):
    """The central-difference gradient of `f` along `axis` (every axis, by default).

    One array per axis, or a single array when only one axis is differentiated. `varargs` gives
    a uniform spacing per differentiated axis (one value, or one per axis); non-uniform
    coordinate arrays are not supported.
    """
    if edge_order != 1:
        raise NotImplementedError("np.gradient only supports edge_order=1")
    f = np.asanyarray(f)
    if axis is None:
        axes = tuple(range(f.ndim))
    elif isinstance(axis, (int, np.integer)):
        axes = (int(axis) % f.ndim,)
    else:
        axes = tuple(int(ax) % f.ndim for ax in axis)

    if len(varargs) == 0:
        spacings = [1.0] * len(axes)
    elif len(varargs) == 1:
        spacings = [varargs[0]] * len(axes)
    elif len(varargs) == len(axes):
        spacings = list(varargs)
    else:
        raise TypeError("invalid number of arguments")

    dtype = f.dtype if f.dtype.kind in "fc" else np.float64
    results = []
    for ax, spacing in zip(axes, spacings):
        if not isinstance(spacing, (int, float, np.number)):
            raise NotImplementedError("np.gradient only supports scalar, uniform spacing")
        n = f.shape[ax]
        result = np.empty(f.shape, dtype=dtype)
        if n == 1:
            result[...] = 0.0
            results.append(result)
            continue
        center, lower, upper = [slice(None)] * f.ndim, [slice(None)] * f.ndim, [slice(None)] * f.ndim
        center[ax], lower[ax], upper[ax] = slice(1, -1), slice(0, -2), slice(2, None)
        result[tuple(center)] = (f[tuple(upper)].astype(dtype) - f[tuple(lower)].astype(dtype)) / 2.0
        first, second = [slice(None)] * f.ndim, [slice(None)] * f.ndim
        first[ax], second[ax] = 0, 1
        result[tuple(first)] = f[tuple(second)].astype(dtype) - f[tuple(first)].astype(dtype)
        last, before_last = [slice(None)] * f.ndim, [slice(None)] * f.ndim
        last[ax], before_last[ax] = -1, -2
        result[tuple(last)] = f[tuple(last)].astype(dtype) - f[tuple(before_last)].astype(dtype)
        results.append(result / spacing)
    return results[0] if len(results) == 1 else results


def interp(x, xp, fp, left=None, right=None, period=None):
    """Piecewise-linear interpolation of `x` against the samples (`xp`, `fp`).

    With `period`, `x` and `xp` are reduced modulo the period, `xp` is sorted, and one sample
    from each end is repeated one period away so points near the wrap interpolate across it.
    `left` and `right` are ignored.
    """
    if period is not None:
        if period == 0:
            raise ValueError("period must be a non-zero value")
        period = abs(period)
        x = np.mod(x, period)
        xp = np.mod(xp, period)
        order = np.argsort(xp)
        xp = xp[order]
        fp = np.asanyarray(fp)[order]
        xp = np.concatenate((xp[-1:] - period, xp, xp[0:1] + period))
        fp = np.concatenate((fp[-1:], fp, fp[0:1]))
        left = None
        right = None
    if np.asanyarray(fp).dtype.kind == "c":
        return _compiled_interp_complex(x, xp, fp, left, right)
    return _compiled_interp(x, xp, fp, left, right)


def correlate(a, v, mode="valid"):
    """The cross-correlation of `a` and `v`; complex `v` is conjugated."""
    return _correlate(a, v, mode, True)


def convolve(a, v, mode="full"):
    """The discrete convolution of `a` and `v`: correlation of `a` with `v` reversed."""
    v = np.asanyarray(v)
    return _correlate(a, v[::-1], mode, False)


def polyfit(x, y, deg):
    """The degree-`deg` polynomial's coefficients (highest power first) that least-squares fit
    the points (`x`, `y`), via :func:`numpy.linalg.lstsq` on the Vandermonde matrix of `x`.
    """
    x = np.asanyarray(x, dtype=np.float64)
    y = np.asanyarray(y, dtype=np.float64)
    vander = np.stack([x**power for power in range(deg, -1, -1)], axis=-1)
    coefficients, _, _, _ = np.linalg.lstsq(vander, y, rcond=None)
    return coefficients


def polyval(p, x):
    """The polynomial with coefficients `p` (highest power first) evaluated at `x`."""
    p = np.asanyarray(p)
    result = 0
    for coefficient in p:
        result = result * x + coefficient
    return result


def cross(a, b, axisa=-1, axisb=-1, axisc=-1, axis=None):
    """The 3-vector cross product of `a` and `b` along their last axis (or `axis`)."""
    if axis is not None:
        axisa = axisb = axisc = axis
    a = np.moveaxis(np.asanyarray(a), axisa, -1)
    b = np.moveaxis(np.asanyarray(b), axisb, -1)
    if a.shape[-1] != 3 or b.shape[-1] != 3:
        raise ValueError("incompatible dimensions for cross product (dimension must be 3)")
    ax, ay, az = a[..., 0], a[..., 1], a[..., 2]
    bx, by, bz = b[..., 0], b[..., 1], b[..., 2]
    result = np.stack([ay * bz - az * by, az * bx - ax * bz, ax * by - ay * bx], axis=-1)
    return np.moveaxis(result, -1, axisc)


def vecdot(a, b, axis=-1):
    """The dot product of `a` and `b` along `axis`, conjugating `a` (as for a complex inner product)."""
    a = np.asanyarray(a)
    b = np.asanyarray(b)
    if a.shape[axis] != b.shape[axis]:
        raise ValueError(
            "vecdot: core dimension mismatch, with gufunc signature (n),(n)->() "
            f"(size {b.shape[axis]} is different from {a.shape[axis]})"
        )
    left = np.conjugate(a) if a.dtype.kind == "c" else a
    return np.sum(left * b, axis=axis)


def select(condlist, choicelist, default=0):
    """`choicelist[i]` at positions where `condlist[i]` is the first true condition, else `default`."""
    if len(condlist) != len(choicelist):
        raise ValueError("list of cases must be same length as list of conditions")
    if len(condlist) == 0:
        raise ValueError("select with an empty condition list is not possible")
    for index, cond in enumerate(condlist):
        if np.asanyarray(cond).dtype != np.bool_:
            raise TypeError(f"invalid entry {index} in condlist: should be boolean ndarray")
    result = np.asanyarray(default)
    for cond, choice in zip(reversed(condlist), reversed(choicelist)):
        result = np.where(np.asanyarray(cond), choice, result)
    return result


def extract(condition, arr):
    """The elements of (raveled) `arr` at the positions where (raveled) `condition` is nonzero."""
    condition = np.asanyarray(condition)
    arr = np.asanyarray(arr)
    return arr.reshape(-1)[np.flatnonzero(condition.reshape(-1))]


def _require_safe_cast(source_dtype, dest_dtype):
    """Raise as NumPy does when `place`/`putmask` are given a real array of values: unlike a
    plain boolean-mask assignment (which casts permissively), both functions insist the value
    array's dtype cast to the destination's dtype under the 'safe' rule.
    """
    if not np.can_cast(source_dtype, dest_dtype, casting="safe"):
        raise TypeError(
            f"Cannot cast array data from {source_dtype!r} to {dest_dtype!r} according to the rule 'safe'"
        )


def place(arr, mask, vals):
    """Write `vals`, cycled, into `arr` at the positions where `mask` is true, in C order."""
    if not isinstance(arr, np.ndarray):
        raise TypeError(f"argument 1 must be numpy.ndarray, not {type(arr).__name__}")
    mask = np.asanyarray(mask, dtype=np.bool_)
    if mask.size != arr.size:
        raise ValueError("mask and data must be the same size")
    mask = mask.reshape(arr.shape)
    count = int(np.count_nonzero(mask))
    if count == 0:
        return None
    if isinstance(vals, np.ndarray):
        pool = vals.reshape(-1)
        if pool.size == 0:
            raise ValueError("Cannot insert from an empty array!")
        _require_safe_cast(pool.dtype, arr.dtype)
        cycled = np.take(pool, np.arange(count) % pool.size)
    else:
        pool = list(vals)
        if len(pool) == 0:
            raise ValueError("Cannot insert from an empty array!")
        cycled = [pool[i % len(pool)] for i in range(count)]
    arr[mask] = cycled
    return None


def putmask(a, mask, values):
    """Write `values`, cycled by flat position (not by the count of true positions), where `mask` is true."""
    if not isinstance(a, np.ndarray):
        raise TypeError("putmask: first argument must be an array")
    mask = np.asanyarray(mask, dtype=np.bool_)
    if mask.size != a.size:
        raise ValueError("putmask: mask and data must be the same size")
    if not a.flags.writeable:
        raise ValueError("putmask: output array is read-only")
    mask = mask.reshape(a.shape)
    positions = np.flatnonzero(mask.reshape(-1))
    if isinstance(values, np.ndarray):
        pool = values.reshape(-1)
        if pool.size == 0:
            return None
        _require_safe_cast(pool.dtype, a.dtype)
        selected = np.take(pool, positions % pool.size)
    else:
        pool = values if isinstance(values, (list, tuple)) else [values]
        if len(pool) == 0:
            return None
        selected = [pool[int(index) % len(pool)] for index in positions]
    a[mask] = selected
    return None


_DIMENSION = r"\w+"
_DIMENSION_LIST = rf"(?:{_DIMENSION}(?:,{_DIMENSION})*)?"
_ARGUMENT = rf"\({_DIMENSION_LIST}\)"
_ARGUMENT_LIST = rf"{_ARGUMENT}(?:,{_ARGUMENT})*"
_SIGNATURE = re.compile(rf"^{_ARGUMENT_LIST}->{_ARGUMENT_LIST}$")
_ARGUMENT_DIMS = re.compile(r"\(([^)]*)\)")


def _parse_signature(signature):
    """A ``(input_dims, output_dims)`` pair, each a list of dimension-name tuples per argument."""
    text = signature.replace(" ", "")
    if not _SIGNATURE.match(text):
        raise ValueError(f"not a valid gufunc signature: {signature}")
    inputs_text, outputs_text = text.split("->")
    parse = lambda part: [
        tuple(name for name in dims.split(",") if name) for dims in _ARGUMENT_DIMS.findall(part)
    ]
    return parse(inputs_text), parse(outputs_text)


def _parse_otypes(otypes):
    """One dtype per output, from a type-code string or a list of dtype-likes."""
    if isinstance(otypes, str):
        resolved = []
        for code in otypes:
            try:
                resolved.append(np.dtype(code))
            except TypeError:
                raise ValueError(f"Invalid otype specified: {code}") from None
        return resolved
    try:
        return [np.dtype(one) for one in otypes]
    except TypeError:
        raise ValueError("Invalid otype specification") from None


class vectorize:
    """A callable that applies `pyfunc` element by element over its array arguments.

    Used directly (``np.vectorize(f)``) or as a decorator, including with only keyword
    arguments (``@np.vectorize(otypes=[float])``), in which case the decorated function is
    filled in as `pyfunc` on the next call.

    Without a ``signature=``, every non-excluded argument is broadcast together and the function
    runs once per broadcast position on plain Python scalars (``ndarray.tolist()``'s elements,
    not NumPy scalars, matching NumPy's own object-array loop). When `otypes` is not given, one
    extra "trial" call on NumPy-scalar inputs at position zero discovers the output dtype(s)
    before the real loop runs, exactly duplicating NumPy's own (mildly wasteful) behavior.

    With a ``signature=``, a gufunc-style ``"(n),(m)->(k)"`` string, the leading, non-core
    dimensions broadcast instead, and the function runs once per broadcast position on the
    core-shaped slices, matching :func:`numpy.core.sum`-style reductions applied along the last
    axes.
    """

    def __init__(self, pyfunc=None, otypes=None, doc=None, excluded=None, cache=False, signature=None):
        self.pyfunc = pyfunc
        self.otypes = _parse_otypes(otypes) if otypes is not None else None
        self.excluded = set(excluded) if excluded is not None else set()
        self.signature = signature
        self._core_dims = _parse_signature(signature) if signature is not None else None
        self.__doc__ = doc if doc is not None else getattr(pyfunc, "__doc__", None)
        if pyfunc is not None:
            self.__name__ = getattr(pyfunc, "__name__", "vectorize")

    def __call__(self, *args, **kwargs):
        if self.pyfunc is None:
            (func,) = args
            return vectorize(
                func,
                otypes=self.otypes,
                doc=self.__doc__,
                excluded=self.excluded or None,
                signature=self.signature,
            )
        if self._core_dims is not None:
            return self._call_with_signature(args, kwargs)
        return self._call_elementwise(args, kwargs)

    def _call_elementwise(self, args, kwargs):
        included = [(index, np.asanyarray(value)) for index, value in enumerate(args) if index not in self.excluded]
        if not included:
            raise TypeError("vectorize needs at least one non-excluded argument")
        shape = np.broadcast_shapes(*(value.shape for _, value in included))
        size = 1
        for dim in shape:
            size *= dim

        if self.otypes is None:
            if size == 0:
                raise ValueError("cannot call `vectorize` on size 0 inputs")
            trial_args = list(args)
            for index, value in included:
                trial_args[index] = np.broadcast_to(value, shape).reshape(-1)[0]
            trial_result = self.pyfunc(*trial_args, **kwargs)
            multiple = isinstance(trial_result, tuple)
            outputs = trial_result if multiple else (trial_result,)
            otypes = [np.asanyarray(one).dtype for one in outputs]
        else:
            otypes = self.otypes
            multiple = len(otypes) > 1

        if size == 0:
            empty = [np.empty(shape, dtype=dtype) for dtype in otypes]
            return tuple(empty) if multiple else empty[0]

        flat_args = list(args)
        for index, value in included:
            flat_args[index] = np.broadcast_to(value, shape).reshape(-1).tolist()

        collected = [[] for _ in otypes]
        for position in range(size):
            call_args = list(flat_args)
            for index, _ in included:
                call_args[index] = flat_args[index][position]
            result = self.pyfunc(*call_args, **kwargs)
            values = result if multiple else (result,)
            for slot, value in zip(collected, values):
                slot.append(value)

        outputs = [np.array(values, dtype=dtype).reshape(shape) for values, dtype in zip(collected, otypes)]
        return tuple(outputs) if multiple else outputs[0]

    def _call_with_signature(self, args, kwargs):
        input_dims, output_dims = self._core_dims
        if len(args) != len(input_dims):
            raise TypeError(f"wrong number of positional arguments: expected {len(input_dims)}, got {len(args)}")
        arrays = [np.asanyarray(value) for value in args]
        outer_shapes = [
            array.shape[: array.ndim - len(dims)] if dims else array.shape
            for array, dims in zip(arrays, input_dims)
        ]
        outer_shape = np.broadcast_shapes(*outer_shapes)
        size = 1
        for dim in outer_shape:
            size *= dim

        flat_arrays = []
        for array, dims in zip(arrays, input_dims):
            core_shape = array.shape[array.ndim - len(dims) :] if dims else ()
            broadcast = np.broadcast_to(array, outer_shape + core_shape)
            flat_arrays.append(broadcast.reshape((size,) + core_shape))

        output_count = len(output_dims)
        collected = [[] for _ in range(output_count)]
        for position in range(size):
            call_args = [flat[position] for flat in flat_arrays]
            result = self.pyfunc(*call_args, **kwargs)
            values = result if output_count > 1 else (result,)
            for slot, value in zip(collected, values):
                slot.append(np.asanyarray(value))

        outputs = [
            np.stack(values, axis=0).reshape(outer_shape + values[0].shape) for values in collected
        ]
        return tuple(outputs) if output_count > 1 else outputs[0]
