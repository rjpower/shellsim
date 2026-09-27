"""The ``nan*`` reductions: NumPy's ordinary reductions with NaN excluded from the count instead
of poisoning the result. Each function replaces NaN with an identity value (0 for sums, +/-inf
for extrema) and, where NumPy does, reports an all-NaN slice with the same warning or error NumPy
raises instead of silently returning that identity.
"""

import warnings

import numpy as np

__all__ = [
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
]


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
        from numpy._statistics import median

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
