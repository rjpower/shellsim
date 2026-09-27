"""Shape-changing helpers built on the native ``reshape``, ``concatenate``, ``take`` and
``expand_dims`` primitives: stacking, splitting, tiling, rolling, and ``kron``/``block``.

``kron`` and ``block`` build their result through broadcasting and ``concatenate`` rather than a
Python loop over elements, so their cost is charged by the array primitives they call, and the
result they allocate is reserved up front the same way any other array constructor's is.
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
    "column_stack",
    "dsplit",
    "dstack",
    "fliplr",
    "flipud",
    "hsplit",
    "hstack",
    "kron",
    "normalize_axis_index",
    "normalize_axis_tuple",
    "put_along_axis",
    "roll",
    "row_stack",
    "split",
    "stack",
    "take_along_axis",
    "tile",
    "vsplit",
    "vstack",
]


def normalize_axis_index(axis, ndim, msg_prefix=None):
    """A single axis, normalized to ``0 <= axis < ndim``; raises ``AxisError`` if out of range."""
    return _normalize_axis_index(axis, ndim, msg_prefix)


def normalize_axis_tuple(axis, ndim, argname=None, allow_duplicate=False):
    """`axis` as a tuple of normalized, non-negative axis indices."""
    axes = (axis,) if isinstance(axis, (int, np.integer)) else tuple(axis)
    normalized = tuple(normalize_axis_index(a, ndim) for a in axes)
    if not allow_duplicate and len(set(normalized)) != len(normalized):
        if argname:
            raise ValueError(f"repeated axis in `{argname}` argument")
        raise ValueError("repeated axis")
    return normalized


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
