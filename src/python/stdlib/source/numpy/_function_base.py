"""Elementwise and signal helpers that NumPy writes in Python on top of ufuncs.

``round``, ``around`` and ``clip`` follow ``numpy/_core/fromnumeric.py``; ``convolve``,
``correlate``, ``cross``, ``argwhere`` and ``flatnonzero`` follow ``numpy/_core/numeric.py``;
``nan_to_num`` and the complex-type predicates follow ``numpy/lib/_type_check_impl.py``;
``angle``, ``gradient`` and ``interp`` follow ``numpy/lib/_function_base_impl.py``; and
``isposinf``, ``isneginf`` and ``fix`` follow ``numpy/lib/_ufunclike_impl.py``; and
``polyval`` and ``polyfit`` follow ``numpy/lib/_polynomial_impl.py``. The native kernels behind
``interp`` and ``correlate`` live in ``_numpy_math``.
"""

import warnings

from _numpy import (
    absolute,
    arctan2,
    array,
    asanyarray,
    asarray,
    complexfloating,
    copyto,
    empty,
    empty_like,
    float64,
    inexact,
    integer,
    isinf,
    isnan,
    issubdtype,
    logical_and,
    multiply,
    nonzero,
    ones,
    pi,
    promote_types,
    signbit,
    sqrt,
    trunc,
    zeros,
    zeros_like,
)
from _numpy import complex128 as _complex128
from _numpy import ndim as _ndim
from _numpy_math import _compiled_interp, _compiled_interp_complex, _correlate
from _numpy_products import dot, outer
from _numpy_shape import (
    _normalize_axis_index,
    broadcast_shapes,
    concatenate,
    moveaxis,
    ravel,
    transpose,
)
from numpy._getlimits import finfo
from numpy._shape_base import atleast_1d, diff, normalize_axis_tuple, vander

_NoValue = object()


def _wrapit(obj, method, *args, **kwds):
    arr = asarray(obj)
    return getattr(arr, method)(*args, **kwds)


def _wrapfunc(obj, method, *args, **kwds):
    bound = getattr(obj, method, None)
    if bound is None:
        return _wrapit(obj, method, *args, **kwds)
    try:
        return bound(*args, **kwds)
    except TypeError:
        # An object with a differently shaped method of this name goes through an array.
        return _wrapit(obj, method, *args, **kwds)


def round(a, decimals=0, out=None):
    return _wrapfunc(a, "round", decimals=decimals, out=out)


def around(a, decimals=0, out=None):
    return _wrapfunc(a, "round", decimals=decimals, out=out)


def clip(a, a_min=_NoValue, a_max=_NoValue, out=None, *, min=_NoValue, max=_NoValue, **kwargs):
    if a_min is _NoValue and a_max is _NoValue:
        a_min = None if min is _NoValue else min
        a_max = None if max is _NoValue else max
    elif a_min is _NoValue:
        raise TypeError("clip() missing 1 required positional argument: 'a_min'")
    elif a_max is _NoValue:
        raise TypeError("clip() missing 1 required positional argument: 'a_max'")
    elif min is not _NoValue or max is not _NoValue:
        raise ValueError(
            "Passing `min` or `max` keyword argument when `a_min` and `a_max` are provided is "
            "forbidden."
        )
    return _wrapfunc(a, "clip", a_min, a_max, out=out, **kwargs)


def argwhere(a):
    if _ndim(a) == 0:
        a = atleast_1d(a)
        return argwhere(a)[:, :0]
    return transpose(nonzero(a))


def flatnonzero(a):
    return nonzero(ravel(a))[0]


def correlate(a, v, mode="valid"):
    return _correlate(a, v, mode, True)


def convolve(a, v, mode="full"):
    a, v = array(a, copy=None, ndmin=1), array(v, copy=None, ndmin=1)
    if len(a) == 0:
        raise ValueError("a cannot be empty")
    if len(v) == 0:
        raise ValueError("v cannot be empty")
    if len(v) > len(a):
        a, v = v, a
    return _correlate(a, v[::-1], mode, False)


def cross(a, b, axisa=-1, axisb=-1, axisc=-1, axis=None):
    if axis is not None:
        axisa, axisb, axisc = (axis,) * 3
    a = asarray(a)
    b = asarray(b)
    if a.ndim < 1 or b.ndim < 1:
        raise ValueError("At least one array has zero dimension")
    axisa = _normalize_axis_index(axisa, a.ndim, msg_prefix="axisa")
    axisb = _normalize_axis_index(axisb, b.ndim, msg_prefix="axisb")
    a = moveaxis(a, axisa, -1)
    b = moveaxis(b, axisb, -1)
    if a.shape[-1] != 3 or b.shape[-1] != 3:
        raise ValueError(
            "Both input arrays must be (arrays of) 3-dimensional vectors, but they are "
            f"{a.shape[-1]} and {b.shape[-1]} dimensional instead."
        )
    shape = (*broadcast_shapes(a.shape[:-1], b.shape[:-1]), 3)
    axisc = _normalize_axis_index(axisc, len(shape), msg_prefix="axisc")
    dtype = promote_types(a.dtype, b.dtype)
    cp = empty(shape, dtype)
    a = a.astype(dtype)
    b = b.astype(dtype)
    a0 = a[..., 0]
    a1 = a[..., 1]
    a2 = a[..., 2]
    b0 = b[..., 0]
    b1 = b[..., 1]
    b2 = b[..., 2]
    cp0 = cp[..., 0]
    cp1 = cp[..., 1]
    cp2 = cp[..., 2]
    multiply(a1, b2, out=cp0)
    tmp = multiply(a2, b1, out=...)
    cp0 -= tmp
    multiply(a2, b0, out=cp1)
    multiply(a0, b2, out=tmp)
    cp1 -= tmp
    multiply(a0, b1, out=cp2)
    multiply(a1, b0, out=tmp)
    cp2 -= tmp
    return moveaxis(cp, -1, axisc)


def real(val):
    try:
        return val.real
    except AttributeError:
        return asanyarray(val).real


def imag(val):
    try:
        return val.imag
    except AttributeError:
        return asanyarray(val).imag


def iscomplex(x):
    ax = asanyarray(x)
    if issubclass(ax.dtype.type, complexfloating):
        return ax.imag != 0
    res = zeros(ax.shape, bool)
    return res[()]


def isreal(x):
    return imag(x) == 0


def iscomplexobj(x):
    try:
        type_ = x.dtype.type
    except AttributeError:
        type_ = asarray(x).dtype.type
    return issubclass(type_, complexfloating)


def isrealobj(x):
    return not iscomplexobj(x)


def real_if_close(a, tol=100):
    a = asanyarray(a)
    type_ = a.dtype.type
    if not issubclass(type_, complexfloating):
        return a
    if tol > 1:
        tol = finfo(type_).eps * tol
    if (absolute(a.imag) < tol).all():
        a = a.real
    return a


def nan_to_num(x, copy=True, nan=0.0, posinf=None, neginf=None):
    x = array(x, subok=True, copy=copy)
    xtype = x.dtype.type
    isscalar = x.ndim == 0
    if not issubclass(xtype, inexact):
        return x[()] if isscalar else x
    iscomplex = issubclass(xtype, complexfloating)
    dest = (x.real, x.imag) if iscomplex else (x,)
    limits = finfo(x.real.dtype)
    maxf, minf = limits.max, limits.min
    if posinf is not None:
        maxf = posinf
    if neginf is not None:
        minf = neginf
    for d in dest:
        idx_nan = isnan(d)
        idx_posinf = isposinf(d)
        idx_neginf = isneginf(d)
        copyto(d, nan, where=idx_nan)
        copyto(d, maxf, where=idx_posinf)
        copyto(d, minf, where=idx_neginf)
    return x[()] if isscalar else x


def isposinf(x, out=None):
    is_inf = isinf(x)
    try:
        positive = ~signbit(x)
    except TypeError as e:
        dtype = asanyarray(x).dtype
        raise TypeError(
            f"This operation is not supported for {dtype} values because it would be ambiguous."
        ) from e
    return logical_and(is_inf, positive, out)


def isneginf(x, out=None):
    is_inf = isinf(x)
    try:
        negative = signbit(x)
    except TypeError as e:
        dtype = asanyarray(x).dtype
        raise TypeError(
            f"This operation is not supported for {dtype} values because it would be ambiguous."
        ) from e
    return logical_and(is_inf, negative, out)


def fix(x, out=None):
    warnings.warn(
        "numpy.fix is deprecated. Use numpy.trunc instead, which is faster and follows the "
        "Array API standard.",
        DeprecationWarning,
        stacklevel=2,
    )
    return trunc(x, out=out)


def angle(z, deg=False):
    z = asanyarray(z)
    if issubclass(z.dtype.type, complexfloating):
        zimag = z.imag
        zreal = z.real
    else:
        zimag = 0
        zreal = z
    a = arctan2(zimag, zreal)
    if deg:
        a *= 180 / pi
    return a


def interp(x, xp, fp, left=None, right=None, period=None):
    fp = asarray(fp)
    if iscomplexobj(fp):
        interp_func = _compiled_interp_complex
        input_dtype = _complex128
    else:
        interp_func = _compiled_interp
        input_dtype = float64
    if period is not None:
        if period == 0:
            raise ValueError("period must be a non-zero value")
        period = abs(period)
        left = None
        right = None
        x = asarray(x, dtype=float64)
        xp = asarray(xp, dtype=float64)
        fp = asarray(fp, dtype=input_dtype)
        if xp.ndim != 1 or fp.ndim != 1:
            raise ValueError("Data points must be 1-D sequences")
        if xp.shape[0] != fp.shape[0]:
            raise ValueError("fp and xp are not of the same length")
        # Normalize both coordinates to one period and extend the samples by one period on
        # each side, so points near the wrap-around interpolate across it.
        x = x % period
        xp = xp % period
        asort_xp = xp.argsort()
        xp = xp[asort_xp]
        fp = fp[asort_xp]
        xp = concatenate((xp[-1:] - period, xp, xp[0:1] + period))
        fp = concatenate((fp[-1:], fp, fp[0:1]))
    return interp_func(x, xp, fp, left, right)


def gradient(f, *varargs, axis=None, edge_order=1):
    f = asanyarray(f)
    N = f.ndim
    if axis is None:
        axes = tuple(range(N))
    else:
        axes = normalize_axis_tuple(axis, N)
    len_axes = len(axes)
    n = len(varargs)
    if n == 0:
        dx = [1.0] * len_axes
    elif n == 1 and _ndim(varargs[0]) == 0:
        dx = varargs * len_axes
    elif n == len_axes:
        dx = list(varargs)
        for i, distances in enumerate(dx):
            distances = asanyarray(distances)
            if distances.ndim == 0:
                continue
            elif distances.ndim != 1:
                raise ValueError("distances must be either scalars or 1d")
            if len(distances) != f.shape[axes[i]]:
                raise ValueError(
                    "when 1d, distances must match the length of the corresponding dimension"
                )
            if issubdtype(distances.dtype, integer):
                distances = distances.astype(float64)
            diffx = diff(distances)
            if (diffx == diffx[0]).all():
                diffx = diffx[0]
            dx[i] = diffx
    else:
        raise TypeError("invalid number of arguments")
    if edge_order > 2:
        raise ValueError("'edge_order' greater than 2 not supported")
    outvals = []
    slice1 = [slice(None)] * N
    slice2 = [slice(None)] * N
    slice3 = [slice(None)] * N
    slice4 = [slice(None)] * N
    otype = f.dtype
    if not issubdtype(otype, inexact):
        # Integer data differentiates in float64.
        if issubdtype(otype, integer):
            f = f.astype(float64)
        otype = float64
    for axis, ax_dx in zip(axes, dx):
        if f.shape[axis] < edge_order + 1:
            raise ValueError(
                "Shape of array too small to calculate a numerical gradient, at least "
                "(edge_order + 1) elements are required."
            )
        out = empty_like(f, dtype=otype)
        uniform_spacing = _ndim(ax_dx) == 0
        slice1[axis] = slice(1, -1)
        slice2[axis] = slice(None, -2)
        slice3[axis] = slice(1, -1)
        slice4[axis] = slice(2, None)
        if uniform_spacing:
            out[tuple(slice1)] = (f[tuple(slice4)] - f[tuple(slice2)]) / (2.0 * ax_dx)
        else:
            dx1 = ax_dx[0:-1]
            dx2 = ax_dx[1:]
            a = -dx2 / (dx1 * (dx1 + dx2))
            b = (dx2 - dx1) / (dx1 * dx2)
            c = dx1 / (dx2 * (dx1 + dx2))
            shape = ones(N, dtype=int)
            shape[axis] = -1
            a = a.reshape(shape)
            b = b.reshape(shape)
            c = c.reshape(shape)
            out[tuple(slice1)] = a * f[tuple(slice2)] + b * f[tuple(slice3)] + c * f[tuple(slice4)]
        if edge_order == 1:
            slice1[axis] = 0
            slice2[axis] = 1
            slice3[axis] = 0
            dx_0 = ax_dx if uniform_spacing else ax_dx[0]
            out[tuple(slice1)] = (f[tuple(slice2)] - f[tuple(slice3)]) / dx_0
            slice1[axis] = -1
            slice2[axis] = -1
            slice3[axis] = -2
            dx_n = ax_dx if uniform_spacing else ax_dx[-1]
            out[tuple(slice1)] = (f[tuple(slice2)] - f[tuple(slice3)]) / dx_n
        else:
            slice1[axis] = 0
            slice2[axis] = 0
            slice3[axis] = 1
            slice4[axis] = 2
            if uniform_spacing:
                a = -1.5 / ax_dx
                b = 2.0 / ax_dx
                c = -0.5 / ax_dx
            else:
                dx1 = ax_dx[0]
                dx2 = ax_dx[1]
                a = -(2.0 * dx1 + dx2) / (dx1 * (dx1 + dx2))
                b = (dx1 + dx2) / (dx1 * dx2)
                c = -dx1 / (dx2 * (dx1 + dx2))
            out[tuple(slice1)] = a * f[tuple(slice2)] + b * f[tuple(slice3)] + c * f[tuple(slice4)]
            slice1[axis] = -1
            slice2[axis] = -3
            slice3[axis] = -2
            slice4[axis] = -1
            if uniform_spacing:
                a = 0.5 / ax_dx
                b = -2.0 / ax_dx
                c = 1.5 / ax_dx
            else:
                dx1 = ax_dx[-2]
                dx2 = ax_dx[-1]
                a = dx2 / (dx1 * (dx1 + dx2))
                b = -(dx2 + dx1) / (dx1 * dx2)
                c = (2.0 * dx2 + dx1) / (dx2 * (dx1 + dx2))
            out[tuple(slice1)] = a * f[tuple(slice2)] + b * f[tuple(slice3)] + c * f[tuple(slice4)]
        outvals.append(out)
        slice1[axis] = slice(None)
        slice2[axis] = slice(None)
        slice3[axis] = slice(None)
        slice4[axis] = slice(None)
    if len_axes == 1:
        return outvals[0]
    return tuple(outvals)


def polyval(p, x):
    p = asarray(p)
    x = asanyarray(x)
    y = zeros_like(x)
    for pv in p:
        y = y * x + pv
    return y


def polyfit(x, y, deg, rcond=None, full=False, w=None, cov=False):
    # Imported here because `numpy.linalg` imports the `numpy` package this module helps build.
    from numpy.exceptions import RankWarning
    from numpy.linalg import inv, lstsq

    order = int(deg) + 1
    x = asarray(x) + 0.0
    y = asarray(y) + 0.0
    if deg < 0:
        raise ValueError("expected deg >= 0")
    if x.ndim != 1:
        raise TypeError("expected 1D vector for x")
    if x.size == 0:
        raise TypeError("expected non-empty vector for x")
    if y.ndim < 1 or y.ndim > 2:
        raise TypeError("expected 1D or 2D array for y")
    if x.shape[0] != y.shape[0]:
        raise TypeError("expected x and y to have same length")
    if rcond is None:
        rcond = len(x) * finfo(x.dtype).eps
    lhs = vander(x, order)
    rhs = y
    if w is not None:
        w = asarray(w) + 0.0
        if w.ndim != 1:
            raise TypeError("expected a 1-d array for weights")
        if w.shape[0] != y.shape[0]:
            raise TypeError("expected w and y to have the same length")
        lhs *= w[:, None]
        if rhs.ndim == 2:
            rhs *= w[:, None]
        else:
            rhs *= w
    scale = sqrt((lhs * lhs).sum(axis=0))
    lhs /= scale
    c, resids, rank, s = lstsq(lhs, rhs, rcond)
    c = (c.T / scale).T
    if rank != order and not full:
        msg = "Polyfit may be poorly conditioned"
        warnings.warn(msg, RankWarning, stacklevel=2)
    if full:
        return (c, resids, rank, s, rcond)
    elif cov:
        Vbase = inv(dot(lhs.T, lhs))
        Vbase /= outer(scale, scale)
        if cov == "unscaled":
            fac = 1
        else:
            if len(x) <= order:
                raise ValueError(
                    "the number of data points must exceed order to scale the covariance matrix"
                )
            fac = resids / (len(x) - order)
        if y.ndim == 1:
            return (c, Vbase * fac)
        else:
            return (c, Vbase[:, :, None] * fac)
    else:
        return c
