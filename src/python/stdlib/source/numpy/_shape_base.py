"""Shape functions that NumPy writes in Python on top of its C primitives.

Stacking, splitting, ``tile``, ``roll``, the triangle helpers, ``vander``, ``diff``, ``rot90``,
``append``, ``resize`` and the ``*_along_axis`` functions follow ``numpy/_core/shape_base.py``,
``numpy/lib/_shape_base_impl.py``, ``numpy/_core/numeric.py``,
``numpy/lib/_twodim_base_impl.py`` and ``numpy/lib/_function_base_impl.py``, so views, copies
and dtypes match NumPy.
"""

import itertools
import operator

from _numpy import (
    arange,
    array,
    asanyarray,
    asarray,
    empty,
    empty_like,
    greater_equal,
    int8,
    int16,
    int32,
    int64,
    integer,
    intp,
    issubdtype,
    multiply,
    ndarray,
    not_equal,
    promote_types,
    subtract,
    where,
    zeros,
    zeros_like,
)
from _numpy import bool as _bool
from _numpy_shape import (
    _normalize_axis_index,
    broadcast_arrays,
    broadcast_to,
    concatenate,
    expand_dims,
    flip,
    ravel,
    reshape,
    swapaxes,
    transpose,
)


def normalize_axis_tuple(axis, ndim, argname=None, allow_duplicate=False):
    """``numpy.lib.array_utils.normalize_axis_tuple``."""
    if type(axis) not in (tuple, list):
        try:
            axis = [operator.index(axis)]
        except TypeError:
            pass
    axis = tuple(_normalize_axis_index(ax, ndim, argname) for ax in axis)
    if not allow_duplicate and len(set(axis)) != len(axis):
        if argname:
            raise ValueError(f"repeated axis in `{argname}` argument")
        raise ValueError("repeated axis")
    return axis


def atleast_1d(*arys):
    if len(arys) == 1:
        result = asanyarray(arys[0])
        if result.ndim == 0:
            result = result.reshape(1)
        return result
    res = []
    for ary in arys:
        result = asanyarray(ary)
        if result.ndim == 0:
            result = result.reshape(1)
        res.append(result)
    return tuple(res)


def atleast_2d(*arys):
    res = []
    for ary in arys:
        result = asanyarray(ary)
        if result.ndim == 0:
            result = result.reshape(1, 1)
        elif result.ndim == 1:
            result = result[None, :]
        res.append(result)
    if len(res) == 1:
        return res[0]
    return tuple(res)


def atleast_3d(*arys):
    res = []
    for ary in arys:
        result = asanyarray(ary)
        if result.ndim == 0:
            result = result.reshape(1, 1, 1)
        elif result.ndim == 1:
            result = result[None, :, None]
        elif result.ndim == 2:
            result = result[:, :, None]
        res.append(result)
    if len(res) == 1:
        return res[0]
    return tuple(res)


def _arrays_for_stack(arrays):
    # shellsim's builtin containers do not expose `__getitem__` as an attribute, so they are
    # recognized by type; other objects follow NumPy's `hasattr` check.
    if not isinstance(arrays, (list, tuple, str, dict, ndarray)) and not hasattr(
        arrays, "__getitem__"
    ):
        raise TypeError(
            'arrays to stack must be passed as a "sequence" type such as list or tuple.'
        )
    return tuple(arrays)


def vstack(tup, *, dtype=None, casting="same_kind"):
    arrs = atleast_2d(*_arrays_for_stack(tup))
    if not isinstance(arrs, tuple):
        arrs = (arrs,)
    return concatenate(arrs, 0, dtype=dtype, casting=casting)


def hstack(tup, *, dtype=None, casting="same_kind"):
    arrs = atleast_1d(*_arrays_for_stack(tup))
    if not isinstance(arrs, tuple):
        arrs = (arrs,)
    if arrs and arrs[0].ndim == 1:
        return concatenate(arrs, 0, dtype=dtype, casting=casting)
    return concatenate(arrs, 1, dtype=dtype, casting=casting)


def stack(arrays, axis=0, out=None, *, dtype=None, casting="same_kind"):
    arrays = [asanyarray(arr) for arr in _arrays_for_stack(arrays)]
    if not arrays:
        raise ValueError("need at least one array to stack")
    shapes = {arr.shape for arr in arrays}
    if len(shapes) != 1:
        raise ValueError("all input arrays must have the same shape")
    result_ndim = arrays[0].ndim + 1
    axis = _normalize_axis_index(axis, result_ndim)
    sl = (slice(None),) * axis + (None,)
    expanded_arrays = [arr[sl] for arr in arrays]
    return concatenate(expanded_arrays, axis=axis, out=out, dtype=dtype, casting=casting)


def dstack(tup):
    arrs = atleast_3d(*_arrays_for_stack(tup))
    if not isinstance(arrs, tuple):
        arrs = (arrs,)
    return concatenate(arrs, 2)


def column_stack(tup):
    arrays = []
    for v in _arrays_for_stack(tup):
        arr = asanyarray(v)
        if arr.ndim < 2:
            arr = array(arr, copy=None, subok=True, ndmin=2).T
        arrays.append(arr)
    return concatenate(arrays, 1)


def array_split(ary, indices_or_sections, axis=0):
    try:
        Ntotal = ary.shape[axis]
    except AttributeError:
        Ntotal = len(ary)
    try:
        Nsections = len(indices_or_sections) + 1
        div_points = [0] + list(indices_or_sections) + [Ntotal]
    except TypeError:
        Nsections = int(indices_or_sections)
        if Nsections <= 0:
            raise ValueError("number sections must be larger than 0.") from None
        Neach_section, extras = divmod(Ntotal, Nsections)
        section_sizes = (
            [0] + extras * [Neach_section + 1] + (Nsections - extras) * [Neach_section]
        )
        div_points = array(section_sizes, dtype=intp).cumsum()
    sub_arys = []
    sary = swapaxes(ary, axis, 0)
    for i in range(Nsections):
        st = div_points[i]
        end = div_points[i + 1]
        sub_arys.append(swapaxes(sary[st:end], axis, 0))
    return sub_arys


def split(ary, indices_or_sections, axis=0):
    try:
        len(indices_or_sections)
    except TypeError:
        sections = indices_or_sections
        N = ary.shape[axis]
        if N % sections:
            raise ValueError("array split does not result in an equal division") from None
    return array_split(ary, indices_or_sections, axis)


def hsplit(ary, indices_or_sections):
    if asanyarray(ary).ndim == 0:
        raise ValueError("hsplit only works on arrays of 1 or more dimensions")
    if ary.ndim > 1:
        return split(ary, indices_or_sections, 1)
    return split(ary, indices_or_sections, 0)


def vsplit(ary, indices_or_sections):
    if asanyarray(ary).ndim < 2:
        raise ValueError("vsplit only works on arrays of 2 or more dimensions")
    return split(ary, indices_or_sections, 0)


def dsplit(ary, indices_or_sections):
    if asanyarray(ary).ndim < 3:
        raise ValueError("dsplit only works on arrays of 3 or more dimensions")
    return split(ary, indices_or_sections, 2)


def tile(A, reps):
    try:
        tup = tuple(reps)
    except TypeError:
        tup = (reps,)
    d = len(tup)
    if all(x == 1 for x in tup) and isinstance(A, ndarray):
        # An array tiled once in every dimension is still copied.
        return array(A, copy=True, subok=True, ndmin=d)
    c = array(A, copy=None, subok=True, ndmin=d)
    if d < c.ndim:
        tup = (1,) * (c.ndim - d) + tup
    shape_out = tuple(s * t for s, t in zip(c.shape, tup))
    n = c.size
    if n > 0:
        for dim_in, nrep in zip(c.shape, tup):
            if nrep != 1:
                c = c.reshape(-1, n).repeat(nrep, 0)
            n //= dim_in
    return c.reshape(shape_out)


def roll(a, shift, axis=None):
    a = asanyarray(a)
    if axis is None:
        return roll(a.ravel(), shift, 0).reshape(a.shape)
    axis = normalize_axis_tuple(axis, a.ndim, allow_duplicate=True)
    shifts_array, axes_array = broadcast_arrays(asanyarray(shift), asanyarray(axis))
    if shifts_array.ndim > 1:
        raise ValueError("'shift' and 'axis' should be scalars or 1D sequences")
    shifts = {ax: 0 for ax in range(a.ndim)}
    for sh, ax in zip(ravel(shifts_array).tolist(), ravel(axes_array).tolist()):
        shifts[ax] += int(sh)
    rolls = [((slice(None), slice(None)),)] * a.ndim
    for ax, offset in shifts.items():
        offset %= a.shape[ax] or 1  # An empty axis has nothing to roll.
        if offset:
            rolls[ax] = (
                (slice(None, -offset), slice(offset, None)),
                (slice(-offset, None), slice(None, offset)),
            )
    result = empty_like(a)
    for indices in itertools.product(*rolls):
        arr_index, res_index = zip(*indices)
        result[res_index] = a[arr_index]
    return result


def _min_int(low, high):
    """The smallest signed integer type holding ``low`` and ``high``."""
    if -128 <= low and high <= 127:
        return int8
    if -32768 <= low and high <= 32767:
        return int16
    if -2147483648 <= low and high <= 2147483647:
        return int32
    return int64


def tri(N, M=None, k=0, dtype=float, *, like=None):
    if M is None:
        M = N
    m = greater_equal.outer(
        arange(N, dtype=_min_int(0, N)), arange(-k, M - k, dtype=_min_int(-k, M - k))
    )
    return m.astype(dtype, copy=False)


def tril(m, k=0):
    m = asanyarray(m)
    mask = tri(*m.shape[-2:], k=k, dtype=_bool)
    return where(mask, m, zeros(1, m.dtype))


def triu(m, k=0):
    m = asanyarray(m)
    mask = tri(*m.shape[-2:], k=k - 1, dtype=_bool)
    return where(mask, zeros(1, m.dtype), m)


def vander(x, N=None, increasing=False):
    x = asarray(x)
    if x.ndim != 1:
        raise ValueError("x must be a one-dimensional array or sequence.")
    if N is None:
        N = len(x)
    v = empty((len(x), N), dtype=promote_types(x.dtype, int))
    tmp = v[:, ::-1] if not increasing else v
    if N > 0:
        tmp[:, 0] = 1
    if N > 1:
        tmp[:, 1:] = x[:, None]
        multiply.accumulate(tmp[:, 1:], out=tmp[:, 1:], axis=1)
    return v


_NoValue = object()


def diff(a, n=1, axis=-1, prepend=_NoValue, append=_NoValue):
    if n == 0:
        return a
    if n < 0:
        raise ValueError("order must be non-negative but got " + repr(n))
    a = asanyarray(a)
    nd = a.ndim
    if nd == 0:
        raise ValueError("diff requires input that is at least one dimensional")
    axis = _normalize_axis_index(axis, nd)
    combined = []
    if prepend is not _NoValue:
        prepend = asanyarray(prepend)
        if prepend.ndim == 0:
            shape = list(a.shape)
            shape[axis] = 1
            prepend = broadcast_to(prepend, tuple(shape))
        combined.append(prepend)
    combined.append(a)
    if append is not _NoValue:
        append = asanyarray(append)
        if append.ndim == 0:
            shape = list(a.shape)
            shape[axis] = 1
            append = broadcast_to(append, tuple(shape))
        combined.append(append)
    if len(combined) > 1:
        a = concatenate(combined, axis)
    slice1 = [slice(None)] * nd
    slice2 = [slice(None)] * nd
    slice1[axis] = slice(1, None)
    slice2[axis] = slice(None, -1)
    slice1 = tuple(slice1)
    slice2 = tuple(slice2)
    op = not_equal if a.dtype == _bool else subtract
    for _ in range(n):
        a = op(a[slice1], a[slice2])
    return a


def flipud(m):
    m = asanyarray(m)
    if m.ndim < 1:
        raise ValueError("Input must be >= 1-d.")
    return m[::-1, ...]


def fliplr(m):
    m = asanyarray(m)
    if m.ndim < 2:
        raise ValueError("Input must be >= 2-d.")
    return m[:, ::-1]


def rot90(m, k=1, axes=(0, 1)):
    axes = tuple(axes)
    if len(axes) != 2:
        raise ValueError("len(axes) must be 2.")
    m = asanyarray(m)
    if axes[0] == axes[1] or abs(axes[0] - axes[1]) == m.ndim:
        raise ValueError("Axes must be different.")
    if axes[0] >= m.ndim or axes[0] < -m.ndim or axes[1] >= m.ndim or axes[1] < -m.ndim:
        raise ValueError(f"Axes={axes} out of range for array of ndim={m.ndim}.")
    k %= 4
    if k == 0:
        return m[:]
    if k == 2:
        return flip(flip(m, axes[0]), axes[1])
    axes_list = list(range(m.ndim))
    axes_list[axes[0]], axes_list[axes[1]] = axes_list[axes[1]], axes_list[axes[0]]
    if k == 1:
        return transpose(flip(m, axes[1]), axes_list)
    return flip(transpose(m, axes_list), axes[1])


def append(arr, values, axis=None):
    arr = asanyarray(arr)
    if axis is None:
        if arr.ndim != 1:
            arr = arr.ravel()
        values = ravel(values)
        axis = arr.ndim - 1
    return concatenate((arr, values), axis=axis)


def resize(a, new_shape):
    if isinstance(new_shape, (int, integer)):
        new_shape = (new_shape,)
    a = ravel(a)
    new_size = 1
    for dim_length in new_shape:
        new_size *= dim_length
        if dim_length < 0:
            raise ValueError("all elements of `new_shape` must be non-negative")
    if a.size == 0 or new_size == 0:
        # An empty source must zero fill; an empty target would repeat nothing.
        return zeros_like(a, shape=new_shape)
    repeats = -(-new_size // a.size)
    a = concatenate((a,) * repeats)[:new_size]
    return reshape(a, new_shape)


def _make_along_axis_idx(arr_shape, indices, axis):
    # compute dimensions to iterate over
    if not issubdtype(indices.dtype, integer):
        raise IndexError("`indices` must be an integer array")
    if len(arr_shape) != indices.ndim:
        raise ValueError("`indices` and `arr` must have the same number of dimensions")
    shape_ones = (1,) * indices.ndim
    dest_dims = list(range(axis)) + [None] + list(range(axis + 1, indices.ndim))

    # build a fancy index, consisting of orthogonal aranges, with the
    # requested index inserted at the right location
    fancy_index = []
    for dim, n in zip(dest_dims, arr_shape):
        if dim is None:
            fancy_index.append(indices)
        else:
            ind_shape = shape_ones[:dim] + (-1,) + shape_ones[dim + 1 :]
            fancy_index.append(arange(n).reshape(ind_shape))

    return tuple(fancy_index)


def take_along_axis(arr, indices, axis=-1):
    if axis is None:
        if indices.ndim != 1:
            raise ValueError("when axis=None, `indices` must have a single dimension.")
        arr = ravel(arr).copy()
        axis = 0
    else:
        axis = _normalize_axis_index(axis, arr.ndim)
    return arr[_make_along_axis_idx(arr.shape, indices, axis)]


def put_along_axis(arr, indices, values, axis):
    if axis is None:
        if indices.ndim != 1:
            raise ValueError("when axis=None, `indices` must have a single dimension.")
        # NumPy assigns into a flattened copy here, so the input is left unchanged.
        arr = ravel(arr).copy()
        axis = 0
    else:
        axis = _normalize_axis_index(axis, arr.ndim)
    arr[_make_along_axis_idx(arr.shape, indices, axis)] = values


def apply_along_axis(func1d, axis, arr, *args, **kwargs):
    from numpy._index_tricks import ndindex

    arr = asanyarray(arr)
    nd = arr.ndim
    axis = _normalize_axis_index(axis, nd)

    # arr, with the iteration axis at the end
    in_dims = list(range(nd))
    inarr_view = transpose(arr, in_dims[:axis] + in_dims[axis + 1 :] + [axis])

    # compute indices for the iteration axes, and append a trailing ellipsis to
    # prevent 0d arrays decaying to scalars, which fixes gh-8642
    inds = ndindex(inarr_view.shape[:-1])
    inds = (ind + (Ellipsis,) for ind in inds)

    # invoke the function on the first item
    try:
        ind0 = next(inds)
    except StopIteration:
        raise ValueError("Cannot apply_along_axis when any iteration dimensions are 0") from None
    res = asanyarray(func1d(inarr_view[ind0], *args, **kwargs))

    # build a buffer for storing evaluations of func1d.
    # remove the requested axis, and add the new ones on the end.
    # laid out so that each write is contiguous.
    # for a tuple index inds, buff[inds] = func1d(inarr_view[inds])
    buff = zeros_like(res, shape=inarr_view.shape[:-1] + res.shape)

    # permutation of axes such that out = buff.transpose(buff_permute)
    buff_dims = list(range(buff.ndim))
    buff_permute = (
        buff_dims[0:axis]
        + buff_dims[buff.ndim - res.ndim : buff.ndim]
        + buff_dims[axis : buff.ndim - res.ndim]
    )

    # save the first result, then compute and save all remaining results
    buff[ind0] = res
    for ind in inds:
        buff[ind] = asanyarray(func1d(inarr_view[ind], *args, **kwargs))

    return transpose(buff, buff_permute)


def apply_over_axes(func, a, axes):
    val = asarray(a)
    N = a.ndim
    if array(axes).ndim == 0:
        axes = (axes,)
    for axis in axes:
        if axis < 0:
            axis = N + axis
        args = (val, axis)
        res = func(*args)
        if res.ndim == val.ndim:
            val = res
        else:
            res = expand_dims(res, axis)
            if res.ndim == val.ndim:
                val = res
            else:
                raise ValueError("function is not returning an array of the correct shape")
    return val
