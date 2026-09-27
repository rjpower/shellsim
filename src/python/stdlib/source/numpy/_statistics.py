"""``median``, ``percentile`` and ``quantile``: NumPy's default ``method="linear"``, which sorts
each reduced slice and linearly interpolates between the two order statistics closest to the
requested rank. ``average`` (a weighted mean) and ``cov``/``corrcoef`` (second-moment statistics
between variables) live here too, next to the other whole-array-in, small-result-out statistics.
"""

import numpy as np
from numpy._methods import _count_reduce_items
from numpy._methods import _mean as mean
from numpy._methods import _std as std
from numpy._methods import _var as var

__all__ = ["average", "corrcoef", "cov", "mean", "median", "percentile", "quantile", "std", "var"]


def _rank_axes(ndim, axis):
    """The reduced axes as a sorted, non-negative tuple, and the axes that remain."""
    if axis is None:
        axes = tuple(range(ndim))
    elif isinstance(axis, (int, np.integer)):
        axes = (int(axis) % ndim,)
    else:
        axes = tuple(int(ax) % ndim for ax in axis)
    keep = tuple(ax for ax in range(ndim) if ax not in axes)
    return axes, keep


def _quantile(a, q, axis, keepdims, kind):
    a = np.asanyarray(a)
    if a.dtype.kind not in "biuf":
        raise TypeError("a must be an array of real numbers")
    q = np.asanyarray(q, dtype=np.float64)
    if kind == "percentile":
        if bool(np.any(q < 0)) or bool(np.any(q > 100)):
            raise ValueError("Percentiles must be in the range [0, 100]")
        q = q / 100.0
    else:
        if bool(np.any(q < 0)) or bool(np.any(q > 1)):
            raise ValueError("Quantiles must be in the range [0, 1]")

    axes, keep = _rank_axes(a.ndim, axis)
    moved = np.moveaxis(a, axes, tuple(range(a.ndim - len(axes), a.ndim)))
    keep_shape = moved.shape[: a.ndim - len(axes)]
    count = 1
    for ax in axes:
        count *= a.shape[ax]
    flat = moved.reshape(keep_shape + (count,)).astype(np.float64)
    sorted_values = np.sort(flat, axis=-1)

    rank = (count - 1) * q
    lo = np.floor(rank).astype(np.int64)
    frac = rank - lo
    hi = np.minimum(lo + 1, count - 1)
    lo_values = np.take(sorted_values, lo, axis=-1)
    hi_values = np.take(sorted_values, hi, axis=-1)
    result = lo_values + frac * (hi_values - lo_values)
    # `lo`/`hi` indexing appended `q`'s shape after `keep_shape`; NumPy puts the quantile axes
    # first instead.
    result = np.moveaxis(result, tuple(range(len(keep_shape), result.ndim)), tuple(range(q.ndim)))

    if result.ndim == 0:
        result = result[()]
    elif keepdims:
        for ax in sorted(axes):
            result = np.expand_dims(result, axis=ax + q.ndim)
    return result


def _check_method(method, kwargs):
    if method != "linear":
        raise NotImplementedError(f"np.percentile/quantile method={method!r} is not supported")
    if kwargs:
        raise NotImplementedError(f"unsupported keyword arguments {sorted(kwargs)}")


def median(a, axis=None, out=None, overwrite_input=False, keepdims=False):
    """The middle value of `a` (average of the two middle values when `a` has even size)."""
    result = _quantile(a, 0.5, axis=axis, keepdims=keepdims, kind="quantile")
    if out is not None:
        out[...] = result
        return out
    return result


def percentile(a, q, axis=None, out=None, overwrite_input=False, method="linear", keepdims=False, **kwargs):
    """The `q`-th percentile(s) (0-100) of `a`, linearly interpolated between order statistics."""
    _check_method(method, kwargs)
    result = _quantile(a, q, axis=axis, keepdims=keepdims, kind="percentile")
    if out is not None:
        out[...] = result
        return out
    return result


def quantile(a, q, axis=None, out=None, overwrite_input=False, method="linear", keepdims=False, **kwargs):
    """The `q`-th quantile(s) (0-1) of `a`, linearly interpolated between order statistics."""
    _check_method(method, kwargs)
    result = _quantile(a, q, axis=axis, keepdims=keepdims, kind="quantile")
    if out is not None:
        out[...] = result
        return out
    return result


def average(a, axis=None, weights=None, returned=False, *, keepdims=False):
    """The mean of `a`, or the weighted mean when `weights` is given."""
    a = np.asanyarray(a)
    if weights is None:
        result = np.mean(a, axis=axis, keepdims=keepdims)
        if not returned:
            return result
        count = _count_reduce_items(a, axis)
        weight_sum = np.full(np.shape(result), count, dtype=np.asanyarray(result).dtype)
        return result, weight_sum[()] if weight_sum.ndim == 0 else weight_sum

    weights = np.asanyarray(weights, dtype=np.float64)
    if weights.shape != a.shape:
        if axis is None:
            raise TypeError("Axis must be specified when shapes of a and weights differ.")
        if weights.ndim != 1:
            raise TypeError("1D weights expected when shapes of a and weights differ.")
        if weights.shape[0] != a.shape[axis]:
            raise ValueError("Length of weights not compatible with specified axis.")
        shape = [1] * a.ndim
        shape[axis] = weights.shape[0]
        weights = np.broadcast_to(weights.reshape(shape), a.shape)

    weight_sum = np.sum(weights, axis=axis, keepdims=keepdims)
    if bool(np.any(weight_sum == 0)):
        raise ZeroDivisionError("Weights sum to zero, can't be normalized")
    result = np.sum(a * weights, axis=axis, keepdims=keepdims) / weight_sum
    if returned:
        return result, weight_sum + np.zeros_like(result)
    return result


def cov(m, y=None, rowvar=True, bias=False, ddof=None):
    """The covariance matrix between the variables in `m` (rows, unless `rowvar` is false)."""
    m = np.asanyarray(m)
    dtype = np.result_type(m.dtype, np.float64)
    m = np.atleast_2d(m.astype(dtype))
    if not rowvar and m.shape[0] != 1:
        m = m.T
    if y is not None:
        y = np.atleast_2d(np.asanyarray(y).astype(dtype))
        if not rowvar and y.shape[0] != 1:
            y = y.T
        m = np.concatenate([m, y], axis=0)
    if ddof is None:
        ddof = 0 if bias else 1
    deviations = m - np.mean(m, axis=1, keepdims=True)
    normalization = m.shape[1] - ddof
    result = (deviations @ deviations.T) / normalization
    return result.squeeze()


def corrcoef(m, y=None, rowvar=True):
    """`cov` normalized so each variable has unit variance: Pearson correlation coefficients."""
    c = cov(m, y, rowvar)
    if c.ndim < 2:
        return np.ones_like(c)
    spread = np.sqrt(np.diag(c))
    result = c / spread[:, None] / spread[None, :]
    return np.clip(result, -1.0, 1.0)
