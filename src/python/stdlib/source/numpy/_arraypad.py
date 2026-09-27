"""``np.pad``: extend an array's edges under a boundary mode.

Every mode processes one axis at a time, growing the array with ``concatenate`` or ``take``, so
padding a later axis sees the already-padded array and fills its corners consistently with the
earlier axes. ``reflect``, ``symmetric`` and ``wrap`` compute the source index for every padded
position with a triangle-wave (reflect/symmetric) or modulo (wrap) formula, so a single
``np.take`` builds an axis's whole padded run, however wide, without a Python loop over elements.
"""

import numpy as np

__all__ = ["pad"]

_INDEX_MODES = ("reflect", "symmetric", "wrap")


def _as_pairs(value, ndim):
    """`value` broadcast to one ``(before, after)`` pair per axis, NumPy's own pad-argument rule."""
    arr = np.asarray(value)
    if arr.ndim == 0:
        return [(arr.item(), arr.item())] * ndim
    if arr.ndim == 1:
        values = arr.tolist()
        if len(values) == 1:
            return [(values[0], values[0])] * ndim
        if len(values) == 2:
            return [(values[0], values[1])] * ndim
        raise ValueError("could not broadcast pad width/values to the array's dimensions")
    if arr.ndim == 2:
        rows = arr.tolist()
        if len(rows) == 1:
            rows = rows * ndim
        if len(rows) != ndim or any(len(row) != 2 for row in rows):
            raise ValueError("could not broadcast pad width/values to the array's dimensions")
        return [(row[0], row[1]) for row in rows]
    raise ValueError("could not broadcast pad width/values to the array's dimensions")


def _block_shape(array, axis, size):
    shape = list(array.shape)
    shape[axis] = size
    return tuple(shape)


def _pad_constant_axis(array, axis, before, after, before_value, after_value):
    parts = []
    if before:
        parts.append(np.full(_block_shape(array, axis, before), before_value, dtype=array.dtype))
    parts.append(array)
    if after:
        parts.append(np.full(_block_shape(array, axis, after), after_value, dtype=array.dtype))
    return np.concatenate(parts, axis=axis) if len(parts) > 1 else array


def _pad_edge_axis(array, axis, before, after):
    parts = []
    if before:
        parts.append(np.repeat(np.take(array, [0], axis=axis), before, axis=axis))
    parts.append(array)
    if after:
        last = array.shape[axis] - 1
        parts.append(np.repeat(np.take(array, [last], axis=axis), after, axis=axis))
    return np.concatenate(parts, axis=axis) if len(parts) > 1 else array


def _fold_index(offset, n, mode):
    """The source index in ``[0, n)`` that padded position `offset` (0 is the first real element)
    copies from, for the periodic boundary modes."""
    if mode == "wrap":
        return offset % n
    if mode == "reflect":
        period = 2 * (n - 1)
        folded = offset % period
        return np.where(folded >= n, period - folded, folded)
    period = 2 * n
    folded = offset % period
    return np.where(folded >= n, period - 1 - folded, folded)


def _pad_index_axis(array, axis, before, after, mode):
    n = array.shape[axis]
    if n == 0:
        raise ValueError(f"can't extend empty axis {axis} using modes other than 'constant' or 'empty'")
    if mode == "reflect" and n == 1:
        return _pad_edge_axis(array, axis, before, after)
    offsets = np.arange(-before, n + after)
    indices = _fold_index(offsets, n, mode)
    return np.take(array, indices, axis=axis)


def pad(array, pad_width, mode="constant", **kwargs):
    """Pad `array` by `pad_width` (per axis, before/after) elements under `mode`.

    Supported modes: ``constant`` (fill value from `constant_values`, default 0), ``edge``
    (repeat the border), and ``reflect``/``symmetric``/``wrap`` (mirror or tile the data; the two
    mirror modes differ in whether the edge value itself repeats).
    """
    array = np.asanyarray(array)
    widths = _as_pairs(pad_width, array.ndim)
    for before, after in widths:
        if before < 0 or after < 0:
            raise ValueError("index can't contain negative values")

    if mode == "constant":
        values = _as_pairs(kwargs.pop("constant_values", 0), array.ndim)
        if kwargs:
            raise TypeError(f"pad() got unexpected keyword arguments {sorted(kwargs)}")
        result = array
        for axis, ((before, after), (before_value, after_value)) in enumerate(zip(widths, values)):
            result = _pad_constant_axis(result, axis, int(before), int(after), before_value, after_value)
        return result

    if mode == "edge":
        if kwargs:
            raise TypeError(f"pad() got unexpected keyword arguments {sorted(kwargs)}")
        result = array
        for axis, (before, after) in enumerate(widths):
            result = _pad_edge_axis(result, axis, int(before), int(after))
        return result

    if mode in _INDEX_MODES:
        reflect_type = kwargs.pop("reflect_type", "even")
        if kwargs or reflect_type != "even":
            raise NotImplementedError(f"np.pad(mode={mode!r}) only supports the 'even' reflect_type")
        result = array
        for axis, (before, after) in enumerate(widths):
            result = _pad_index_axis(result, axis, int(before), int(after), mode)
        return result

    raise NotImplementedError(f"np.pad mode {mode!r} is not supported")
