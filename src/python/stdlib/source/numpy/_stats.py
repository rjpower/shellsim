"""Statistics: ``mean``/``var``/``std`` (the implementations the native ``ndarray`` methods call
back into through ``numpy._stats``, and that ``np.mean``/``np.var``/``np.std`` reuse directly),
``median``/``percentile``/``quantile`` (NumPy's default ``method="linear"``, sorting each reduced
slice and linearly interpolating between the two order statistics closest to the requested rank),
``average``, ``cov``/``corrcoef``, the ``histogram`` family, and the ``nan*`` reductions (each
ordinary reduction with NaN excluded from the count instead of poisoning the result).
"""

import warnings

import numpy as np

__all__ = [
    "average",
    "corrcoef",
    "cov",
    "digitize",
    "histogram",
    "histogram_bin_edges",
    "mean",
    "median",
    "nanargmax",
    "nanargmin",
    "nancumprod",
    "nancumsum",
    "nanmax",
    "nanmean",
    "nanmedian",
    "nanmin",
    "nanprod",
    "nanstd",
    "nansum",
    "nanvar",
    "percentile",
    "quantile",
    "std",
    "var",
]


# -- ndarray-method fallbacks (`a.mean()`, `.var()` and `.std()` reach these by name through
# -- `numpy._stats`; see `reduce.rs`'s `python_method`) ------------------------------------------


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


def _count(a, axis, where, keepdims, dtype):
    """The number of elements each output cell of a reduction over `axis` combines: a Python
    int without a mask, or an array of per-cell counts in `dtype` under a `where=` mask."""
    if where is True:
        return _count_reduce_items(a, axis)
    mask = np.broadcast_to(np.asarray(where, dtype=bool), a.shape)
    return np.sum(mask, axis=axis, keepdims=keepdims).astype(dtype)


def _mean(a, axis=None, dtype=None, out=None, keepdims=False, *, where=True):
    a = np.asanyarray(a)
    result_dtype = _result_dtype(a, dtype)
    total = np.sum(a, axis=axis, dtype=result_dtype, keepdims=keepdims, where=where)
    count = _count(a, axis, where, keepdims, np.asarray(total).dtype)
    if np.any(np.asarray(count) == 0):
        warnings.warn("Mean of empty slice", RuntimeWarning, stacklevel=2)
    with np.errstate(invalid="ignore", divide="ignore"):
        result = total / count
    if out is not None:
        out[...] = result
        return out
    return result


def _var(a, axis=None, dtype=None, out=None, ddof=0, keepdims=False, *, where=True):
    a = np.asanyarray(a)
    is_complex = np.issubdtype(a.dtype, np.complexfloating)
    result_dtype = _result_dtype(a, dtype)
    with np.errstate(invalid="ignore", divide="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mean_value = _mean(a, axis=axis, dtype=result_dtype, keepdims=True, where=where)
    if is_complex:
        deviation = a - mean_value
        squared = (deviation * deviation.conjugate()).real
    else:
        deviation = a.astype(result_dtype) - mean_value
        squared = deviation * deviation
    total = np.sum(squared, axis=axis, dtype=squared.dtype, keepdims=keepdims, where=where)
    divisor = _count(a, axis, where, keepdims, np.asarray(total).dtype) - ddof
    if np.any(np.asarray(divisor) <= 0):
        warnings.warn("Degrees of freedom <= 0 for slice", RuntimeWarning, stacklevel=2)
    with np.errstate(invalid="ignore", divide="ignore"):
        result = total / divisor
    if out is not None:
        out[...] = result
        return out
    return result


def _std(a, axis=None, dtype=None, out=None, ddof=0, keepdims=False, *, where=True):
    variance = _var(a, axis=axis, dtype=dtype, ddof=ddof, keepdims=keepdims, where=where)
    result = np.sqrt(variance)
    if out is not None:
        out[...] = result
        return out
    return result


# -- public statistics: mean/var/std, order statistics, moments ------------------------------

mean = _mean
std = _std
var = _var


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


# -- histograms ---------------------------------------------------------------------------------


def histogram_bin_edges(a, bins=10, range=None, weights=None):
    """The edges :func:`histogram` would use for `a`, `bins` and `range`."""
    a = np.asanyarray(a)
    if isinstance(bins, str):
        raise NotImplementedError(f"np.histogram_bin_edges(bins={bins!r}) is not supported")
    bins_array = np.asanyarray(bins)
    if bins_array.ndim == 1:
        edges = bins_array.astype(np.float64)
        if edges.size > 1 and bool(np.any(edges[1:] < edges[:-1])):
            raise ValueError("bins must increase monotonically, when an array")
        return edges
    count = int(bins)
    if count < 1:
        raise ValueError("`bins` must be positive, when an integer")
    if range is not None:
        low, high = float(range[0]), float(range[1])
    elif a.size == 0:
        low, high = 0.0, 1.0
    else:
        low, high = float(np.min(a)), float(np.max(a))
    if low == high:
        low, high = low - 0.5, high + 0.5
    return np.linspace(low, high, count + 1)


def histogram(a, bins=10, range=None, density=False, weights=None):
    """Counts (or, if `density`, a probability density) of `a` over `bins`, and the bin edges."""
    a = np.asanyarray(a).reshape(-1)
    if weights is not None:
        weights = np.asanyarray(weights).reshape(-1)
    edges = histogram_bin_edges(a, bins, range, weights)
    bin_count = edges.size - 1
    positions = np.searchsorted(edges, a, side="right") - 1
    # A value equal to the last edge belongs in the last bin, not past it.
    positions = np.where(a == edges[-1], bin_count - 1, positions)
    in_range = (positions >= 0) & (positions < bin_count)
    kept = positions[in_range]
    if weights is None:
        counts = np.bincount(kept, minlength=bin_count)[:bin_count]
    else:
        counts = np.bincount(kept, weights=weights[in_range], minlength=bin_count)[:bin_count]
    if density:
        widths = np.diff(edges)
        counts = counts / (counts.sum() * widths)
    return counts, edges


def digitize(x, bins, right=False):
    """The index of the bin (from `bins`, increasing or decreasing) each value of `x` falls in."""
    x = np.asanyarray(x)
    bins = np.asanyarray(bins)
    side = "left" if right else "right"
    if bins.size < 2 or bool(bins[-1] >= bins[0]):
        return np.searchsorted(bins, x, side=side)
    return bins.size - np.searchsorted(bins[::-1], x, side=side)


# -- nan-aware reductions -----------------------------------------------------------------------


def _nan_mask(a):
    if a.dtype.kind != "f":
        return np.zeros(a.shape, dtype=np.bool_)
    return np.isnan(a)


def nansum(a, axis=None, dtype=None, out=None, keepdims=False):
    a = np.asanyarray(a)
    filled = np.where(_nan_mask(a), 0, a)
    return np.sum(filled, axis=axis, dtype=dtype, out=out, keepdims=keepdims)


def nanprod(a, axis=None, dtype=None, out=None, keepdims=False):
    a = np.asanyarray(a)
    filled = np.where(_nan_mask(a), 1, a)
    return np.prod(filled, axis=axis, dtype=dtype, out=out, keepdims=keepdims)


def nanmean(a, axis=None, dtype=None, out=None, keepdims=False):
    a = np.asanyarray(a)
    mask = _nan_mask(a)
    filled = np.where(mask, 0, a)
    total = np.sum(filled, axis=axis, dtype=dtype, keepdims=keepdims)
    count = np.sum(~mask, axis=axis, keepdims=keepdims)
    if bool(np.any(count == 0)):
        warnings.warn("Mean of empty slice", RuntimeWarning, stacklevel=2)
    with np.errstate(invalid="ignore"):
        result = total / count
    if out is not None:
        out[...] = result
        return out
    return result


def _nan_extreme(reducer, a, axis, out, keepdims, is_min):
    a = np.asanyarray(a)
    mask = _nan_mask(a)
    if not bool(np.any(mask)):
        return reducer(a, axis=axis, out=out, keepdims=keepdims)
    fill = np.finfo(a.dtype).max if is_min else np.finfo(a.dtype).min
    filled = np.where(mask, fill, a)
    all_nan = np.all(mask, axis=axis, keepdims=keepdims)
    result = np.asanyarray(reducer(filled, axis=axis, keepdims=keepdims))
    if bool(np.any(all_nan)):
        warnings.warn("All-NaN slice encountered", RuntimeWarning, stacklevel=2)
        result = np.where(all_nan, np.float64("nan"), result)
    if axis is None and not keepdims and result.ndim == 0:
        result = result[()]
    if out is not None:
        out[...] = result
        return out
    return result


def nanmin(a, axis=None, out=None, keepdims=False):
    return _nan_extreme(np.min, a, axis, out, keepdims, is_min=True)


def nanmax(a, axis=None, out=None, keepdims=False):
    return _nan_extreme(np.max, a, axis, out, keepdims, is_min=False)


def nanvar(a, axis=None, dtype=None, out=None, ddof=0, keepdims=False):
    a = np.asanyarray(a)
    mask = _nan_mask(a)
    filled = np.where(mask, 0, a)
    with np.errstate(invalid="ignore"):
        mean_value = np.sum(filled, axis=axis, keepdims=True) / np.sum(~mask, axis=axis, keepdims=True)
    deviations = np.where(mask, 0, a - mean_value)
    total = np.sum(deviations * deviations, axis=axis, dtype=dtype, keepdims=keepdims)
    divisor = np.sum(~mask, axis=axis, keepdims=keepdims) - ddof
    with np.errstate(invalid="ignore", divide="ignore"):
        result = total / divisor
    if out is not None:
        out[...] = result
        return out
    return result


def nanstd(a, axis=None, dtype=None, out=None, ddof=0, keepdims=False):
    result = np.sqrt(nanvar(a, axis=axis, dtype=dtype, ddof=ddof, keepdims=keepdims))
    if out is not None:
        out[...] = result
        return out
    return result


def nanmedian(a, axis=None, out=None, overwrite_input=False, keepdims=False):
    a = np.asanyarray(a)
    mask = _nan_mask(a)
    if not bool(np.any(mask)):
        return median(a, axis=axis, out=out, keepdims=keepdims)

    if axis is None:
        axes = tuple(range(a.ndim))
    elif isinstance(axis, (int, np.integer)):
        axes = (int(axis) % a.ndim,)
    else:
        axes = tuple(int(ax) % a.ndim for ax in axis)
    moved = np.moveaxis(a, axes, tuple(range(a.ndim - len(axes), a.ndim)))
    moved_mask = np.moveaxis(mask, axes, tuple(range(a.ndim - len(axes), a.ndim)))
    keep_shape = moved.shape[: a.ndim - len(axes)]
    count = 1
    for ax in axes:
        count *= a.shape[ax]
    flat = moved.reshape(keep_shape + (count,))
    flat_mask = moved_mask.reshape(keep_shape + (count,))

    sorted_values = np.sort(flat, axis=-1)  # NaN sorts last, as in every shellsim/NumPy sort.
    valid = np.sum(~flat_mask, axis=-1)
    rank = (valid.astype(np.float64) - 1) * 0.5
    lo = np.clip(np.floor(rank), 0, None).astype(np.int64)
    hi = np.clip(lo + 1, 0, np.maximum(valid - 1, 0))
    lo_values = np.take_along_axis(sorted_values, lo[..., None], axis=-1)[..., 0]
    hi_values = np.take_along_axis(sorted_values, hi[..., None], axis=-1)[..., 0]
    frac = rank - np.floor(rank)
    with np.errstate(invalid="ignore"):
        result = lo_values + frac * (hi_values - lo_values)

    all_nan = valid == 0
    if bool(np.any(all_nan)):
        warnings.warn("All-NaN slice encountered", RuntimeWarning, stacklevel=2)
        result = np.where(all_nan, np.float64("nan"), result)
    if result.ndim == 0:
        result = result[()]
    elif keepdims:
        for ax in sorted(axes):
            result = np.expand_dims(result, axis=ax)
    if out is not None:
        out[...] = result
        return out
    return result


def _nan_arg(reducer, a, axis, keepdims, is_min):
    a = np.asanyarray(a)
    mask = _nan_mask(a)
    if not bool(np.any(mask)):
        return reducer(a, axis=axis, keepdims=keepdims)
    if bool(np.any(np.all(mask, axis=axis, keepdims=True))):
        raise ValueError("All-NaN slice encountered")
    fill = np.finfo(a.dtype).max if is_min else np.finfo(a.dtype).min
    filled = np.where(mask, fill, a)
    return reducer(filled, axis=axis, keepdims=keepdims)


def nanargmin(a, axis=None, out=None, *, keepdims=False):
    return _nan_arg(np.argmin, a, axis, keepdims, is_min=True)


def nanargmax(a, axis=None, out=None, *, keepdims=False):
    return _nan_arg(np.argmax, a, axis, keepdims, is_min=False)


def nancumsum(a, axis=None, dtype=None, out=None):
    a = np.asanyarray(a)
    filled = np.where(_nan_mask(a), 0, a)
    return np.cumsum(filled, axis=axis, dtype=dtype, out=out)


def nancumprod(a, axis=None, dtype=None, out=None):
    a = np.asanyarray(a)
    filled = np.where(_nan_mask(a), 1, a)
    return np.cumprod(filled, axis=axis, dtype=dtype, out=out)
