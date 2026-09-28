"""Integration of sampled data: ``trapezoid``, ``cumulative_trapezoid`` and ``simpson``.

Each rule works along one axis of an array of samples, spaced either uniformly by ``dx`` or at
the coordinates ``x``. A one-dimensional ``x`` gives the coordinates along the axis; an ``x``
with ``y``'s number of dimensions broadcasts against it. Python-float spacings stay weak in
NumPy's promotion rules, so ``float32`` samples give ``float32`` results.
"""

import numpy as np

trapezoid = np.trapezoid


def _along(ndim, axis, index):
    """A tuple index that applies ``index`` to one axis and takes every other axis whole."""
    key = [slice(None)] * ndim
    key[axis] = index
    return tuple(key)


def _spacing(y, x, axis):
    """The differences of ``x`` along ``axis``, shaped to broadcast against ``y``."""
    x = np.asarray(x)
    if x.ndim == 1:
        shape = [1] * y.ndim
        shape[axis] = -1
        return np.diff(x).reshape(shape)
    if x.ndim != y.ndim:
        raise ValueError("If given, shape of x must be 1-D or the same as y.")
    return np.diff(x, axis=axis)


def cumulative_trapezoid(y, x=None, dx=1.0, axis=-1, initial=None):
    """Running trapezoid-rule integrals of ``y`` along ``axis``.

    The result has one fewer sample along ``axis`` than ``y``, or the same number when
    ``initial=0`` prepends a zero.

    >>> cumulative_trapezoid([1, 2, 3], initial=0).tolist()
    [0.0, 1.5, 4.0]
    """
    y = np.asarray(y)
    if y.shape[axis] == 0:
        raise ValueError("At least one point is required along `axis`.")
    if x is None:
        d = dx
    else:
        d = _spacing(y, x, axis)
        if d.shape[axis] != y.shape[axis] - 1:
            raise ValueError("If given, length of x along axis must be the same as y.")
    upper = y[_along(y.ndim, axis, slice(1, None))]
    lower = y[_along(y.ndim, axis, slice(None, -1))]
    result = np.cumsum(d * (upper + lower) / 2.0, axis=axis)
    if initial is None:
        return result
    if initial != 0:
        raise ValueError("`initial` must be `None` or `0`.")
    shape = list(result.shape)
    shape[axis] = 1
    return np.concatenate([np.zeros(shape, dtype=result.dtype), result], axis=axis)


def _divide(numerator, denominator):
    """``numerator / denominator``, or zero where the denominator is zero, without a warning.

    Coincident sample points give zero-width intervals, which contribute nothing.
    """
    if np.ndim(denominator) == 0:
        return numerator / denominator if denominator != 0 else 0 * numerator
    with np.errstate(divide="ignore", invalid="ignore"):
        quotient = numerator / denominator
    return np.where(denominator != 0, quotient, 0.0)


def _simpson_pairs(y, h, axis):
    """Simpson's rule over an odd number of samples, one parabola per pair of intervals.

    ``h`` is the uniform spacing, or the interval widths along ``axis`` for uneven samples.
    """
    ndim = y.ndim
    y0 = y[_along(ndim, axis, slice(0, -2, 2))]
    y1 = y[_along(ndim, axis, slice(1, -1, 2))]
    y2 = y[_along(ndim, axis, slice(2, None, 2))]
    if np.ndim(h) == 0:
        return h / 3.0 * np.sum(y0 + 4.0 * y1 + y2, axis)
    h0 = h[_along(ndim, axis, slice(0, None, 2))]
    h1 = h[_along(ndim, axis, slice(1, None, 2))]
    h_sum = h0 + h1
    ratio = _divide(h0, h1)
    weighted = (
        y0 * (2.0 - _divide(1.0, ratio))
        + y1 * (h_sum * _divide(h_sum, h0 * h1))
        + y2 * (2.0 - ratio)
    )
    return np.sum(h_sum / 6.0 * weighted, axis)


def simpson(y, x=None, *, dx=1.0, axis=-1):
    """Integrate samples ``y`` along ``axis`` with composite Simpson's rule.

    An odd number of samples is covered by parabolas through consecutive triples. For an even
    number, the parabola through the last three samples also covers the final interval, as in
    Cartwright's correction. Two samples fall back to the trapezoid rule and one gives zero.

    >>> float(simpson([1, 2, 3, 4]))
    7.5
    """
    y = np.asarray(y)
    n = y.shape[axis]
    if x is not None:
        x = np.asarray(x)
        if x.ndim not in (1, y.ndim):
            raise ValueError("If given, shape of x must be 1-D or the same as y.")
        if x.shape[axis if x.ndim > 1 else 0] != n:
            raise ValueError("If given, length of x along axis must be the same as y.")
    if n == 0:
        raise IndexError(f"index -1 is out of bounds for axis {axis % y.ndim} with size 0")
    if n < 3:
        return trapezoid(y, x, dx=dx, axis=axis)
    h = dx if x is None else _spacing(y, x, axis)
    if n % 2 == 1:
        return _simpson_pairs(y, h, axis)
    head = _along(y.ndim, axis, slice(None, -1))
    if np.ndim(h) == 0:
        result = _simpson_pairs(y[head], h, axis)
        h0 = h1 = h
    else:
        result = _simpson_pairs(y[head], h[head], axis)
        h0 = h[_along(y.ndim, axis, -2)]
        h1 = h[_along(y.ndim, axis, -1)]
    alpha = _divide(2 * h1**2 + 3 * h0 * h1, 6 * (h0 + h1))
    beta = _divide(h1**2 + 3 * h0 * h1, 6 * h0)
    eta = _divide(h1**3, 6 * h0 * (h0 + h1))
    last = y[_along(y.ndim, axis, -1)]
    middle = y[_along(y.ndim, axis, -2)]
    first = y[_along(y.ndim, axis, -3)]
    return result + (alpha * last + beta * middle - eta * first)
