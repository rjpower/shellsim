"""Shape-changing helpers built on the native ``reshape``, ``transpose``, ``concatenate``,
``take``, ``diagonal`` and ``broadcast_to`` primitives: stacking, splitting, tiling, rolling,
axis moves and removal, flipping, ``trace``, broadcasting shapes, ``kron``/``block``, and ``pad``
(extending an array's edges under a boundary mode).

``squeeze``, ``expand_dims`` and ``flip`` never move data: dropping or inserting length-one axes
is always representable as a ``reshape`` view, and reversing an axis is ordinary negative-step
slicing. ``moveaxis``/``swapaxes`` are ``transpose`` with a computed axis order. ``trace`` is
``diagonal(...).sum(-1)``, reusing the read-only diagonal view and the ``add`` reduction. ``kron``,
``block`` and ``pad`` build their result through broadcasting, ``concatenate`` and ``take`` rather
than a Python loop over elements, so their cost is charged by the array primitives they call, and
the result they allocate is reserved up front the same way any other array constructor's is.
``reflect``, ``symmetric`` and ``wrap`` padding compute the source index for every padded position
with a triangle-wave (reflect/symmetric) or modulo (wrap) formula, so a single ``np.take`` builds
an axis's whole padded run, however wide, without a Python loop over elements.
"""

import numpy as np
from _numpy_shape import _normalize_axis_index

__all__ = [
    "apply_along_axis",
    "apply_over_axes",
    "array_split",
    "atleast_1d",
    "atleast_2d",
    "atleast_3d",
    "block",
    "broadcast_arrays",
    "broadcast_shapes",
    "column_stack",
    "dsplit",
    "dstack",
    "expand_dims",
    "flip",
    "fliplr",
    "flipud",
    "hsplit",
    "hstack",
    "kron",
    "moveaxis",
    "normalize_axis_index",
    "normalize_axis_tuple",
    "pad",
    "put_along_axis",
    "roll",
    "row_stack",
    "split",
    "squeeze",
    "stack",
    "swapaxes",
    "take_along_axis",
    "tile",
    "trace",
    "vsplit",
    "vstack",
]


def normalize_axis_index(axis, ndim, msg_prefix=None):
    """A single axis, normalized to ``0 <= axis < ndim``; raises ``AxisError`` if out of range."""
    return _normalize_axis_index(axis, ndim, msg_prefix)


def normalize_axis_tuple(axis, ndim, argname=None, allow_duplicate=False):
    """`axis` as a tuple of normalized, non-negative axis indices; each out-of-range axis raises
    `AxisError` naming `argname`, as NumPy's own `normalize_axis_tuple` does."""
    axes = (axis,) if isinstance(axis, (int, np.integer)) else tuple(axis)
    normalized = tuple(normalize_axis_index(a, ndim, argname) for a in axes)
    if not allow_duplicate and len(set(normalized)) != len(normalized):
        if argname:
            raise ValueError(f"repeated axis in `{argname}` argument")
        raise ValueError("repeated axis")
    return normalized


def squeeze(a, axis=None):
    """`a` with the length-one axes named by `axis` removed (every length-one axis, if `None`).
    Dropping axes without reordering the rest is always a valid `reshape`, so this never copies."""
    a = np.asanyarray(a)
    ndim = a.ndim
    if axis is None:
        selected = tuple(ax for ax in range(ndim) if a.shape[ax] == 1)
    elif ndim == 0 and not isinstance(axis, tuple):
        value = int(axis)
        if value not in (0, -1):
            normalize_axis_index(value, ndim)
        selected = ()
    else:
        selected = normalize_axis_tuple(axis, ndim)
        if any(a.shape[ax] != 1 for ax in selected):
            raise ValueError("cannot select an axis to squeeze out which has size not equal to one")
    shape = tuple(a.shape[ax] for ax in range(ndim) if ax not in selected)
    return a.reshape(shape)


def expand_dims(a, axis):
    """`a` with new length-one axes inserted at the (output) positions `axis`."""
    a = np.asanyarray(a)
    count = 1 if isinstance(axis, (int, np.integer)) else len(tuple(axis))
    rank = a.ndim + count
    axes = normalize_axis_tuple(axis, rank)
    source = iter(a.shape)
    shape = tuple(1 if pos in axes else next(source) for pos in range(rank))
    return a.reshape(shape)


def moveaxis(a, source, destination):
    """`a` with the `source` axes moved to `destination`, the rest kept in their relative order."""
    a = np.asanyarray(a)
    ndim = a.ndim
    source = normalize_axis_tuple(source, ndim, "source")
    destination = normalize_axis_tuple(destination, ndim, "destination")
    if len(source) != len(destination):
        raise ValueError("`source` and `destination` arguments must have the same number of elements")
    order = [ax for ax in range(ndim) if ax not in source]
    for dest, src in sorted(zip(destination, source)):
        order.insert(dest, src)
    return a.transpose(order)


def swapaxes(a, axis1, axis2):
    """`a` with `axis1` and `axis2` exchanged."""
    a = np.asanyarray(a)
    ndim = a.ndim
    first = normalize_axis_index(axis1, ndim, "axis1")
    second = normalize_axis_index(axis2, ndim, "axis2")
    order = list(range(ndim))
    order[first], order[second] = order[second], order[first]
    return a.transpose(order)


def flip(m, axis=None):
    """`m` with the elements along `axis` (every axis, if `None`) reversed: a negative-step
    slice, so the result is a view."""
    m = np.asanyarray(m)
    ndim = m.ndim
    if axis is None:
        if ndim == 0:
            return m[()]
        axes = range(ndim)
    else:
        axes = normalize_axis_tuple(axis, ndim)
    index = [slice(None)] * ndim
    for ax in axes:
        index[ax] = slice(None, None, -1)
    return m[tuple(index)]


def trace(a, offset=0, axis1=0, axis2=1, dtype=None, out=None):
    """The sum along diagonal `offset` of the `axis1`/`axis2` plane(s) of `a`."""
    result = np.sum(np.diagonal(a, offset, axis1, axis2), axis=-1, dtype=dtype)
    if out is not None:
        out[...] = result
        return out
    return result


def _common_shape(shapes):
    """The NumPy broadcast shape of `shapes` (each a tuple of non-negative ints)."""
    rank = max((len(shape) for shape in shapes), default=0)
    result = [1] * rank
    source = [0] * rank
    for axis in range(rank):
        for position, shape in enumerate(shapes):
            index = axis + len(shape) - rank
            if index < 0:
                continue
            dimension = shape[index]
            if dimension == 1:
                continue
            if result[axis] == 1:
                result[axis] = dimension
                source[axis] = position
            elif result[axis] != dimension:
                raise ValueError(
                    "shape mismatch: objects cannot be broadcast to a single shape.  Mismatch is "
                    f"between arg {source[axis]} with shape {shapes[source[axis]]} and arg "
                    f"{position} with shape {shape}."
                )
    return tuple(result)


def broadcast_shapes(*args):
    """The shape that broadcasting arrays of shapes `args` together would produce."""
    shapes = []
    for value in args:
        shape = (int(value),) if isinstance(value, (int, np.integer)) else tuple(int(d) for d in value)
        if any(d < 0 for d in shape):
            raise ValueError("negative dimensions are not allowed")
        shapes.append(shape)
    return _common_shape(shapes)


def broadcast_arrays(*args, subok=False):
    """Each of `args` as an array of their common broadcast shape; arrays that already have it
    are returned unchanged, the rest as read-only broadcast views."""
    arrays = [np.asanyarray(a) for a in args]
    shape = _common_shape([a.shape for a in arrays])
    return tuple(a if a.shape == shape else np.broadcast_to(a, shape) for a in arrays)


def _one_atleast(a, ndim, leading):
    a = np.asanyarray(a)
    if a.ndim >= ndim:
        return a
    pad = ndim - a.ndim
    shape = (1,) * pad + a.shape if leading else a.shape + (1,) * pad
    return a.reshape(shape)


def atleast_1d(*arrays):
    results = [_one_atleast(a, 1, leading=True) for a in arrays]
    return results[0] if len(results) == 1 else tuple(results)


def atleast_2d(*arrays):
    results = [_one_atleast(a, 2, leading=True) for a in arrays]
    return results[0] if len(results) == 1 else tuple(results)


def _one_atleast_3d(a):
    a = np.asanyarray(a)
    if a.ndim == 0:
        return a.reshape(1, 1, 1)
    if a.ndim == 1:
        return a.reshape(1, a.shape[0], 1)
    if a.ndim == 2:
        return a.reshape(a.shape + (1,))
    return a


def atleast_3d(*arrays):
    results = [_one_atleast_3d(a) for a in arrays]
    return results[0] if len(results) == 1 else tuple(results)


def vstack(tup, *, dtype=None, casting="same_kind"):
    return np.concatenate([_one_atleast(a, 2, leading=True) for a in tup], axis=0, dtype=dtype, casting=casting)


def hstack(tup, *, dtype=None, casting="same_kind"):
    arrays = [np.asanyarray(a) for a in tup]
    axis = 0 if arrays and arrays[0].ndim == 1 else 1
    return np.concatenate(arrays, axis=axis, dtype=dtype, casting=casting)


row_stack = vstack


def column_stack(tup):
    arrays = [np.asanyarray(a) for a in tup]
    arrays = [a.reshape(-1, 1) if a.ndim == 1 else a for a in arrays]
    return np.concatenate(arrays, axis=1)


def dstack(tup):
    return np.concatenate([_one_atleast_3d(a) for a in tup], axis=2)


def fliplr(m):
    """`m` (at least 2-d) with its columns reversed: a view flipped along axis 1."""
    m = np.asanyarray(m)
    if m.ndim < 2:
        raise ValueError("Input must be >= 2-d.")
    return np.flip(m, axis=1)


def flipud(m):
    """`m` (at least 1-d) with its rows reversed: a view flipped along axis 0."""
    m = np.asanyarray(m)
    if m.ndim < 1:
        raise ValueError("Input must be >= 1-d.")
    return np.flip(m, axis=0)


def stack(arrays, axis=0, out=None, *, dtype=None, casting="same_kind"):
    arrays = [np.asanyarray(a) for a in arrays]
    if not arrays:
        raise ValueError("need at least one array to stack")
    shape = arrays[0].shape
    for a in arrays:
        if a.shape != shape:
            raise ValueError("all input arrays must have the same shape")
    expanded = [np.expand_dims(a, axis) for a in arrays]
    result = np.concatenate(expanded, axis=axis, dtype=dtype, casting=casting)
    if out is not None:
        out[...] = result
        return out
    return result


def array_split(ary, indices_or_sections, axis=0):
    ary = np.asanyarray(ary)
    axis = normalize_axis_index(axis, ary.ndim)
    n = ary.shape[axis]
    if isinstance(indices_or_sections, (int, np.integer)):
        sections = int(indices_or_sections)
        if sections <= 0:
            raise ValueError("number sections must be larger than 0.")
        each, extra = divmod(n, sections)
        sizes = [each + 1] * extra + [each] * (sections - extra)
        edges = [0]
        for size in sizes:
            edges.append(edges[-1] + size)
    else:
        edges = [0, *indices_or_sections, n]
    pieces = []
    for start, stop in zip(edges[:-1], edges[1:]):
        index = [slice(None)] * ary.ndim
        index[axis] = slice(start, stop)
        pieces.append(ary[tuple(index)])
    return pieces


def split(ary, indices_or_sections, axis=0):
    if isinstance(indices_or_sections, (int, np.integer)):
        ary_arr = np.asanyarray(ary)
        axis_norm = normalize_axis_index(axis, ary_arr.ndim)
        sections = int(indices_or_sections)
        if ary_arr.shape[axis_norm] % sections != 0:
            raise ValueError("array split does not result in an equal division")
    return array_split(ary, indices_or_sections, axis)


def hsplit(ary, indices_or_sections):
    ary = np.asanyarray(ary)
    if ary.ndim == 0:
        raise ValueError("hsplit only works on arrays of 1 or more dimensions")
    return split(ary, indices_or_sections, axis=0 if ary.ndim == 1 else 1)


def vsplit(ary, indices_or_sections):
    ary = np.asanyarray(ary)
    if ary.ndim < 2:
        raise ValueError("vsplit only works on arrays of 2 or more dimensions")
    return split(ary, indices_or_sections, axis=0)


def dsplit(ary, indices_or_sections):
    ary = np.asanyarray(ary)
    if ary.ndim < 3:
        raise ValueError("dsplit only works on arrays of 3 or more dimensions")
    return split(ary, indices_or_sections, axis=2)


def tile(A, reps):
    A = np.asanyarray(A)
    reps = (int(reps),) if isinstance(reps, (int, np.integer)) else tuple(int(r) for r in reps)
    d = len(reps)
    if A.ndim < d:
        A = A.reshape((1,) * (d - A.ndim) + A.shape)
    elif A.ndim > d:
        reps = (1,) * (A.ndim - d) + reps
    result = A
    for axis, count in enumerate(reps):
        if count == 1:
            continue
        if count == 0:
            result = np.take(result, np.arange(0), axis=axis)
        else:
            result = np.concatenate([result] * count, axis=axis)
    return result


def roll(a, shift, axis=None):
    a = np.asanyarray(a)
    if axis is None:
        return roll(a.reshape(-1), shift, 0).reshape(a.shape)
    axes = (axis,) if isinstance(axis, (int, np.integer)) else tuple(axis)
    shifts = (shift,) * len(axes) if isinstance(shift, (int, np.integer)) else tuple(shift)
    if len(shifts) != len(axes):
        raise ValueError("shift and axis must have the same number of elements")
    net_shift = {}
    for ax, s in zip(axes, shifts):
        ax = normalize_axis_index(ax, a.ndim)
        net_shift[ax] = net_shift.get(ax, 0) + s
    result = a
    for ax, s in net_shift.items():
        n = result.shape[ax]
        if n == 0:
            continue
        s = s % n
        if s == 0:
            continue
        head = np.take(result, np.arange(n - s, n), axis=ax)
        tail = np.take(result, np.arange(0, n - s), axis=ax)
        result = np.concatenate([head, tail], axis=ax)
    return result


def take_along_axis(arr, indices, axis):
    arr = np.asanyarray(arr)
    indices = np.asanyarray(indices)
    if axis is None:
        arr = arr.reshape(-1)
        indices = indices.reshape(-1)
        axis = 0
    axis = normalize_axis_index(axis, arr.ndim)
    if arr.ndim != indices.ndim:
        raise ValueError("`indices` and `arr` must have the same number of dimensions")
    index_arrays = tuple(
        indices
        if ax == axis
        else np.arange(arr.shape[ax]).reshape([arr.shape[ax] if a == ax else 1 for a in range(arr.ndim)])
        for ax in range(arr.ndim)
    )
    return arr[index_arrays]


def put_along_axis(arr, indices, values, axis):
    indices = np.asanyarray(indices)
    if axis is None:
        arr = arr.reshape(-1)
        indices = indices.reshape(-1)
        axis = 0
    axis = normalize_axis_index(axis, arr.ndim)
    if arr.ndim != indices.ndim:
        raise ValueError("`indices` and `arr` must have the same number of dimensions")
    index_arrays = tuple(
        indices
        if ax == axis
        else np.arange(arr.shape[ax]).reshape([arr.shape[ax] if a == ax else 1 for a in range(arr.ndim)])
        for ax in range(arr.ndim)
    )
    arr[index_arrays] = values


def apply_along_axis(func1d, axis, arr, *args, **kwargs):
    arr = np.asanyarray(arr)
    axis = normalize_axis_index(axis, arr.ndim)
    moved = np.moveaxis(arr, axis, -1)
    outer_shape = moved.shape[:-1]
    count = 1
    for dim in outer_shape:
        count *= dim
    flat = moved.reshape(count, moved.shape[-1])
    results = [np.asanyarray(func1d(flat[i], *args, **kwargs)) for i in range(count)]
    stacked = stack(results, axis=0)
    return stacked.reshape(outer_shape + stacked.shape[1:])


def apply_over_axes(func, a, axes):
    a = np.asanyarray(a)
    axes = (axes,) if isinstance(axes, (int, np.integer)) else axes
    for axis in axes:
        result = np.asanyarray(func(a, axis))
        a = result if result.ndim == a.ndim else np.expand_dims(result, axis)
    return a


def kron(a, b):
    a = np.asanyarray(a)
    b = np.asanyarray(b)
    is_scalar = a.ndim == 0 and b.ndim == 0
    ndim = max(a.ndim, b.ndim, 1)
    if a.ndim < ndim:
        a = a.reshape((1,) * (ndim - a.ndim) + a.shape)
    if b.ndim < ndim:
        b = b.reshape((1,) * (ndim - b.ndim) + b.shape)
    a_expanded = a.reshape([dim for size in a.shape for dim in (size, 1)])
    b_expanded = b.reshape([dim for size in b.shape for dim in (1, size)])
    final_shape = tuple(sa * sb for sa, sb in zip(a.shape, b.shape))
    result = (a_expanded * b_expanded).reshape(final_shape)
    return result.reshape(())[()] if is_scalar else result


def _leaf_depth(node, depth=0):
    if isinstance(node, list) and node:
        return _leaf_depth(node[0], depth + 1)
    return depth


def _check_block(node, path, depth, target_depth):
    if isinstance(node, tuple):
        raise TypeError(
            f"{path} is a tuple. Only lists can be used to arrange blocks, and "
            "np.block does not allow implicit conversion from tuple to ndarray."
        )
    if isinstance(node, list):
        if not node:
            raise ValueError(f"List at {path} cannot be empty")
        for i, item in enumerate(node):
            _check_block(item, f"{path}[{i}]", depth + 1, target_depth)
    elif depth != target_depth:
        raise ValueError(
            f"List depths are mismatched. First element was at depth {target_depth}, "
            f"but there is an element at depth {depth} ({path})"
        )


def _build_block(node, depth, max_depth):
    if not isinstance(node, list):
        arr = np.asanyarray(node)
        target_ndim = max(max_depth, arr.ndim)
        if arr.ndim < target_ndim:
            arr = arr.reshape((1,) * (target_ndim - arr.ndim) + arr.shape)
        return arr
    parts = [_build_block(item, depth + 1, max_depth) for item in node]
    axis = -(max_depth - depth)
    return np.concatenate(parts, axis=axis)


def block(arrays):
    """Assemble an array from nested lists of blocks, concatenated along successive axes."""
    if not isinstance(arrays, list):
        return np.asanyarray(arrays)
    target_depth = _leaf_depth(arrays)
    _check_block(arrays, "arrays", 0, target_depth)
    return _build_block(arrays, 0, target_depth)


_PAD_INDEX_MODES = ("reflect", "symmetric", "wrap")


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


def _pad_block_shape(array, axis, size):
    shape = list(array.shape)
    shape[axis] = size
    return tuple(shape)


def _pad_constant_axis(array, axis, before, after, before_value, after_value):
    parts = []
    if before:
        parts.append(np.full(_pad_block_shape(array, axis, before), before_value, dtype=array.dtype))
    parts.append(array)
    if after:
        parts.append(np.full(_pad_block_shape(array, axis, after), after_value, dtype=array.dtype))
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


def _pad_fold_index(offset, n, mode):
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
    indices = _pad_fold_index(offsets, n, mode)
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

    if mode in _PAD_INDEX_MODES:
        reflect_type = kwargs.pop("reflect_type", "even")
        if kwargs or reflect_type != "even":
            raise NotImplementedError(f"np.pad(mode={mode!r}) only supports the 'even' reflect_type")
        result = array
        for axis, (before, after) in enumerate(widths):
            result = _pad_index_axis(result, axis, int(before), int(after), mode)
        return result

    raise NotImplementedError(f"np.pad mode {mode!r} is not supported")
