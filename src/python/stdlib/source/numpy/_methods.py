"""Python fallbacks for the ``ndarray`` methods that are simplest to express with other array
operations: ``mean``, ``var`` and ``std``. The native ``mean``/``var``/``std`` methods call
``numpy._methods._mean`` etc. with the receiver as the first positional argument, so this module
is reached from both ``a.mean()`` and (through ``numpy._statistics.mean``) ``np.mean(a)``.
"""

import warnings

import numpy as np

__all__ = []


def _count_reduce_items(a, axis):
    """The number of elements a reduction over `axis` (``None`` meaning every axis) combines."""
    if axis is None:
        axes = range(a.ndim)
    elif isinstance(axis, (int, np.integer)):
        axes = (axis if axis >= 0 else axis + a.ndim,)
    else:
        axes = tuple(ax if ax >= 0 else ax + a.ndim for ax in axis)
    count = 1
    for ax in axes:
        count *= a.shape[ax]
    return count


def _result_dtype(a, dtype):
    if dtype is not None:
        return np.dtype(dtype)
    return a.dtype if np.issubdtype(a.dtype, np.inexact) else np.dtype(np.float64)


def _mean(a, axis=None, dtype=None, out=None, keepdims=False):
    a = np.asanyarray(a)
    result_dtype = _result_dtype(a, dtype)
    total = np.sum(a, axis=axis, dtype=result_dtype, keepdims=keepdims)
    count = _count_reduce_items(a, axis)
    if count == 0:
        warnings.warn("Mean of empty slice", RuntimeWarning, stacklevel=2)
        result = total * float("nan")
    else:
        result = total / count
    if out is not None:
        out[...] = result
        return out
    return result


def _var(a, axis=None, dtype=None, out=None, ddof=0, keepdims=False):
    a = np.asanyarray(a)
    is_complex = np.issubdtype(a.dtype, np.complexfloating)
    result_dtype = _result_dtype(a, dtype)
    mean_value = _mean(a, axis=axis, dtype=result_dtype, keepdims=True)
    if is_complex:
        deviation = a - mean_value
        squared = (deviation * deviation.conjugate()).real
    else:
        deviation = a.astype(result_dtype) - mean_value
        squared = deviation * deviation
    count = _count_reduce_items(a, axis)
    divisor = count - ddof
    if divisor <= 0:
        warnings.warn("Degrees of freedom <= 0 for slice", RuntimeWarning, stacklevel=2)
    total = np.sum(squared, axis=axis, dtype=squared.dtype, keepdims=keepdims)
    result = total / divisor
    if out is not None:
        out[...] = result
        return out
    return result


def _std(a, axis=None, dtype=None, out=None, ddof=0, keepdims=False):
    variance = _var(a, axis=axis, dtype=dtype, ddof=ddof, keepdims=keepdims)
    result = np.sqrt(variance)
    if out is not None:
        out[...] = result
        return out
    return result


def _round(a, decimals=0, out=None):
    """``rint(a * 10**decimals) / 10**decimals``, computed at `a`'s own dtype and precision."""
    a = np.asanyarray(a)
    dtype = a.dtype
    if dtype == np.bool_:
        result = a.copy()
    elif np.issubdtype(dtype, np.integer):
        if decimals >= 0:
            result = a.copy()
        else:
            factor = 10.0 ** (-decimals)
            result = (np.rint(a.astype(np.float64) / factor) * factor).astype(dtype)
    else:
        factor = dtype.type(10.0) ** decimals
        result = np.rint(a * factor) / factor
    if a.ndim == 0 and out is None:
        result = result[()]
    if out is not None:
        out[...] = result
        return out
    return result


def _clip(a, min=None, max=None, out=None, **kwargs):
    a = np.asanyarray(a)
    result = a
    if min is not None:
        result = np.maximum(result, min)
    if max is not None:
        result = np.minimum(result, max)
    if result is a:
        result = a.copy()
    if a.ndim == 0 and out is None:
        result = result[()]
    if out is not None:
        out[...] = result
        return out
    return result
