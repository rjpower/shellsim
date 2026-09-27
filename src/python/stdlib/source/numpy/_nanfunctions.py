"""NaN-skipping reductions, following ``numpy/lib/_nanfunctions_impl.py``.

Each function replaces NaN with a neutral value (``0`` for sums, ``1`` for products, ``±inf``
for extrema) and reduces, then restores NaN and warns where a whole slice was NaN. Integer and
boolean arrays cannot hold NaN and go straight to the ordinary reduction.
"""

import warnings

from _numpy import (
    array,
    asanyarray,
    divide,
    fmax,
    fmin,
    inexact,
    intp,
    isnan,
    issubdtype,
    multiply,
    nan,
    ndarray,
    not_equal,
    object_,
    sqrt,
    subtract,
    complexfloating,
)
from _numpy_reduce import all as _all
from _numpy_reduce import any as _any
from _numpy_reduce import argmax, argmin, cumprod, cumsum, prod
from _numpy_reduce import max as _amax
from _numpy_reduce import min as _amin
from _numpy_reduce import sum as _sum
from numpy._errstate import errstate
from numpy._methods import mean as _mean
from numpy._methods import var as _var

_inf = float("inf")


def _replace_nan(a, val):
    a = asanyarray(a)
    if a.dtype == object_:
        mask = not_equal(a, a, dtype=bool)
    elif issubdtype(a.dtype, inexact):
        mask = isnan(a)
    else:
        mask = None
    if mask is not None:
        a = array(a, copy=True)
        a[mask] = val
    return a, mask


def _copyto(a, val, mask):
    if isinstance(a, ndarray):
        a[mask] = val
    else:
        a = a.dtype.type(val)
    return a


def _divide_by_count(a, b, out=None):
    with errstate(invalid="ignore", divide="ignore"):
        if isinstance(a, ndarray):
            return divide(a, b, out=a if out is None else out, casting="unsafe")
        if out is not None:
            return divide(a, b, out=out, casting="unsafe")
        try:
            return a.dtype.type(a / b)
        except AttributeError:
            return a / b


def _require_inexact(dtype, out):
    if dtype is not None and not issubdtype(dtype, inexact):
        raise TypeError("If a is inexact, then dtype must be inexact")
    if out is not None and not issubdtype(out.dtype, inexact):
        raise TypeError("If a is inexact, then out must be inexact")


def _extremum(a, axis, out, keepdims, fast, slow, fill):
    if type(a) is ndarray and a.dtype != object_:
        res = fast.reduce(a, axis=axis, out=out, keepdims=keepdims)
        if _any(isnan(res)):
            warnings.warn("All-NaN slice encountered", RuntimeWarning, stacklevel=3)
        return res
    a, mask = _replace_nan(a, fill)
    res = slow(a, axis=axis, out=out, keepdims=keepdims)
    if mask is None:
        return res
    mask = _all(mask, axis=axis, keepdims=keepdims)
    if _any(mask):
        res = _copyto(res, nan, mask)
        warnings.warn("All-NaN axis encountered", RuntimeWarning, stacklevel=3)
    return res


def nanmin(a, axis=None, out=None, keepdims=False):
    return _extremum(a, axis, out, keepdims, fmin, _amin, _inf)


def nanmax(a, axis=None, out=None, keepdims=False):
    return _extremum(a, axis, out, keepdims, fmax, _amax, -_inf)


def nanargmin(a, axis=None, out=None, *, keepdims=False):
    a, mask = _replace_nan(a, _inf)
    if mask is not None and mask.size and _any(_all(mask, axis=axis)):
        raise ValueError("All-NaN slice encountered")
    return argmin(a, axis=axis, out=out, keepdims=keepdims)


def nanargmax(a, axis=None, out=None, *, keepdims=False):
    a, mask = _replace_nan(a, -_inf)
    if mask is not None and mask.size and _any(_all(mask, axis=axis)):
        raise ValueError("All-NaN slice encountered")
    return argmax(a, axis=axis, out=out, keepdims=keepdims)


def nansum(a, axis=None, dtype=None, out=None, keepdims=False):
    a, _ = _replace_nan(a, 0)
    return _sum(a, axis=axis, dtype=dtype, out=out, keepdims=keepdims)


def nanprod(a, axis=None, dtype=None, out=None, keepdims=False):
    a, _ = _replace_nan(a, 1)
    return prod(a, axis=axis, dtype=dtype, out=out, keepdims=keepdims)


def nancumsum(a, axis=None, dtype=None, out=None):
    a, _ = _replace_nan(a, 0)
    return cumsum(a, axis=axis, dtype=dtype, out=out)


def nancumprod(a, axis=None, dtype=None, out=None):
    a, _ = _replace_nan(a, 1)
    return cumprod(a, axis=axis, dtype=dtype, out=out)


def nanmean(a, axis=None, dtype=None, out=None, keepdims=False):
    arr, mask = _replace_nan(a, 0)
    if mask is None:
        return _mean(arr, axis=axis, dtype=dtype, out=out, keepdims=keepdims)
    _require_inexact(dtype, out)
    cnt = _sum(~mask, axis=axis, dtype=intp, keepdims=keepdims)
    tot = _sum(arr, axis=axis, dtype=dtype, out=out, keepdims=keepdims)
    avg = _divide_by_count(tot, cnt, out=out)
    if _any(cnt == 0):
        warnings.warn("Mean of empty slice", RuntimeWarning, stacklevel=2)
    return avg


def nanvar(a, axis=None, dtype=None, out=None, ddof=0, keepdims=False, *, mean=None,
           correction=None):
    arr, mask = _replace_nan(a, 0)
    if mask is None:
        return _var(arr, axis=axis, dtype=dtype, out=out, ddof=ddof, keepdims=keepdims,
                    mean=mean, correction=correction)
    _require_inexact(dtype, out)
    if correction is not None:
        if ddof != 0:
            raise ValueError("ddof and correction can't be provided simultaneously.")
        ddof = correction
    if mean is not None:
        avg = mean
    else:
        avg = _sum(arr, axis=axis, dtype=dtype, keepdims=True)
        avg = _divide_by_count(avg, _sum(~mask, axis=axis, dtype=intp, keepdims=True))
    subtract(arr, avg, out=arr, casting="unsafe")
    arr = _copyto(arr, 0, mask)
    if issubdtype(arr.dtype, complexfloating):
        sqr = multiply(arr, arr.conj()).real
    else:
        sqr = multiply(arr, arr, out=arr)
    var = _sum(sqr, axis=axis, dtype=dtype, out=out, keepdims=keepdims)
    dof = _sum(~mask, axis=axis, dtype=intp, keepdims=keepdims) - ddof
    var = _divide_by_count(var, dof)
    isbad = dof <= 0
    if _any(isbad):
        warnings.warn("Degrees of freedom <= 0 for slice.", RuntimeWarning, stacklevel=2)
        var = _copyto(var, nan, isbad)
    return var


def nanstd(a, axis=None, dtype=None, out=None, ddof=0, keepdims=False, *, mean=None,
           correction=None):
    var = nanvar(a, axis=axis, dtype=dtype, out=out, ddof=ddof, keepdims=keepdims, mean=mean,
                 correction=correction)
    if isinstance(var, ndarray):
        return sqrt(var, out=var)
    if hasattr(var, "dtype"):
        return var.dtype.type(sqrt(var))
    return sqrt(var)
