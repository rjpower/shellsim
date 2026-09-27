"""NaN-skipping reductions, following ``numpy/lib/_nanfunctions_impl.py``.

Each function replaces NaN with a neutral value (``0`` for sums, ``1`` for products, ``±inf``
for extrema) and reduces, then restores NaN and warns where a whole slice was NaN. Integer and
boolean arrays cannot hold NaN and go straight to the ordinary reduction.

``nanmedian``, ``nanpercentile`` and ``nanquantile`` instead move each slice's NaNs out of the
way and take the ordinary order statistic of the rest. NumPy computes short ``nanmedian`` slices
through masked arrays; here every slice takes the ``apply_along_axis`` path, which gives the same
values.
"""

import warnings

from _numpy import (
    array,
    asanyarray,
    divide,
    empty_like,
    full,
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
from _numpy_shape import moveaxis
from numpy._errstate import errstate
from numpy._methods import mean as _mean
from numpy._index_tricks import ndindex
from numpy._methods import var as _var
from numpy._shape_base import apply_along_axis, normalize_axis_tuple
from numpy._statistics import (
    _quantile_is_valid,
    _quantile_unchecked,
    _ureduce,
    _weights_are_valid,
)

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


def _remove_nan_1d(arr1d, second_arr1d=None, overwrite_input=False):
    """Move the NaNs of a 1-D array to the end and cut them off, doing the same to a second
    array of the same length. Returns the two arrays and whether they may be overwritten."""
    if arr1d.dtype == object_:
        # object arrays do not support `isnan` (gh-9009), so make a guess
        c = not_equal(arr1d, arr1d, dtype=bool)
    else:
        c = isnan(arr1d)
    s = c.nonzero()[0]
    if s.size == arr1d.size:
        warnings.warn("All-NaN slice encountered", RuntimeWarning, stacklevel=6)
        if second_arr1d is None:
            return arr1d[:0], None, True
        return arr1d[:0], second_arr1d[:0], True
    if s.size == 0:
        return arr1d, second_arr1d, overwrite_input
    if not overwrite_input:
        arr1d = arr1d.copy()
    # select non-nans at end of array
    enonan = arr1d[-s.size :][~c[-s.size :]]
    # fill nans in beginning of array with non-nans of end
    arr1d[s[: enonan.size]] = enonan
    if second_arr1d is None:
        return arr1d[: -s.size], None, True
    if not overwrite_input:
        second_arr1d = second_arr1d.copy()
    enonan = second_arr1d[-s.size :][~c[-s.size :]]
    second_arr1d[s[: enonan.size]] = enonan
    return arr1d[: -s.size], second_arr1d[: -s.size], True


def _nanmedian1d(arr1d, overwrite_input=False):
    from numpy._statistics import median

    arr1d_parsed, _, overwrite_input = _remove_nan_1d(arr1d, overwrite_input=overwrite_input)
    if arr1d_parsed.size == 0:
        # Ensure that a nan-esque scalar of the appropriate type (and unit)
        # is returned for `timedelta64` and `complexfloating`
        return arr1d[-1]
    return median(arr1d_parsed, overwrite_input=overwrite_input)


def _nanmedian(a, axis=None, out=None, overwrite_input=False):
    if axis is None or a.ndim == 1:
        part = a.ravel()
        if out is None:
            return _nanmedian1d(part, overwrite_input)
        out[...] = _nanmedian1d(part, overwrite_input)
        return out
    result = apply_along_axis(_nanmedian1d, axis, a, overwrite_input)
    if out is not None:
        out[...] = result
    return result


def nanmedian(a, axis=None, out=None, overwrite_input=False, keepdims=False):
    a = asanyarray(a)
    # apply_along_axis in _nanmedian doesn't handle empty arrays well,
    # so deal them upfront
    if a.size == 0:
        return nanmean(a, axis, out=out, keepdims=keepdims)
    return _ureduce(
        a, func=_nanmedian, keepdims=keepdims, axis=axis, out=out, overwrite_input=overwrite_input
    )


def _nan_check_weights(weights, a, axis, method):
    if method != "inverted_cdf":
        raise ValueError(f"Only method 'inverted_cdf' supports weights. Got: {method}.")
    if axis is not None:
        axis = normalize_axis_tuple(axis, a.ndim, argname="axis")
    weights = _weights_are_valid(weights=weights, a=a, axis=axis)
    if _any(weights < 0):
        raise ValueError("Weights must be non-negative.")
    return weights


def nanpercentile(
    a,
    q,
    axis=None,
    out=None,
    overwrite_input=False,
    method="linear",
    keepdims=False,
    *,
    weights=None,
):
    a = asanyarray(a)
    if a.dtype.kind == "c":
        raise TypeError("a must be an array of real numbers")
    weak_q = type(q) in (int, float)
    q = divide(q, 100, out=...)
    if not _quantile_is_valid(q):
        raise ValueError("Percentiles must be in the range [0, 100]")
    if weights is not None:
        weights = _nan_check_weights(weights, a, axis, method)
    return _nanquantile_unchecked(
        a, q, axis, out, overwrite_input, method, keepdims, weights, weak_q
    )


def nanquantile(
    a,
    q,
    axis=None,
    out=None,
    overwrite_input=False,
    method="linear",
    keepdims=False,
    *,
    weights=None,
):
    a = asanyarray(a)
    if a.dtype.kind == "c":
        raise TypeError("a must be an array of real numbers")
    weak_q = type(q) in (int, float)
    q = asanyarray(q)
    if not _quantile_is_valid(q):
        raise ValueError("Quantiles must be in the range [0, 1]")
    if weights is not None:
        weights = _nan_check_weights(weights, a, axis, method)
    return _nanquantile_unchecked(
        a, q, axis, out, overwrite_input, method, keepdims, weights, weak_q
    )


def _nanquantile_unchecked(
    a,
    q,
    axis=None,
    out=None,
    overwrite_input=False,
    method="linear",
    keepdims=False,
    weights=None,
    weak_q=False,
):
    # apply_along_axis in _nanpercentile doesn't handle empty arrays well,
    # so deal them upfront
    if a.size == 0:
        return nanmean(a, axis, out=out, keepdims=keepdims)
    return _ureduce(
        a,
        func=_nanquantile_ureduce_func,
        q=q,
        weights=weights,
        keepdims=keepdims,
        axis=axis,
        out=out,
        overwrite_input=overwrite_input,
        method=method,
        weak_q=weak_q,
    )


def _nanquantile_ureduce_func(
    a, q, weights, axis=None, out=None, overwrite_input=False, method="linear", weak_q=False
):
    if axis is None or a.ndim == 1:
        part = a.ravel()
        wgt = None if weights is None else weights.ravel()
        result = _nanquantile_1d(part, q, overwrite_input, method, weights=wgt, weak_q=weak_q)
    # Note that this code could try to fill in `out` right away
    elif weights is None:
        result = apply_along_axis(
            _nanquantile_1d, axis, a, q, overwrite_input, method, weights, weak_q
        )
        # apply_along_axis fills in collapsed axis with results.
        # Move those axes to the beginning to match percentile's
        # convention.
        if q.ndim != 0:
            from_ax = [axis + i for i in range(q.ndim)]
            result = moveaxis(result, from_ax, list(range(q.ndim)))
    else:
        # We need to apply along axis over 2 arrays, a and weights.
        # move operation axes to end for simplicity:
        a = moveaxis(a, axis, -1)
        if weights is not None:
            weights = moveaxis(weights, axis, -1)
        if out is not None:
            result = out
        else:
            # weights are limited to `inverted_cdf` so the result dtype
            # is known to be identical to that of `a` here:
            result = empty_like(a, shape=q.shape + a.shape[:-1])
        for ii in ndindex(a.shape[:-1]):
            result[(...,) + ii] = _nanquantile_1d(
                a[ii],
                q,
                weights=weights[ii],
                overwrite_input=overwrite_input,
                method=method,
                weak_q=weak_q,
            )
        return result
    if out is not None:
        out[...] = result
    return result


def _nanquantile_1d(arr1d, q, overwrite_input=False, method="linear", weights=None, weak_q=False):
    """The quantiles of the non-NaN elements of ``arr1d``, or NaN when there are none."""
    arr1d, weights, overwrite_input = _remove_nan_1d(
        arr1d, second_arr1d=weights, overwrite_input=overwrite_input
    )
    if arr1d.size == 0:
        # convert to scalar
        return full(q.shape, nan, dtype=arr1d.dtype)[()]
    return _quantile_unchecked(
        arr1d,
        q,
        overwrite_input=overwrite_input,
        method=method,
        weights=weights,
        weak_q=weak_q,
    )
