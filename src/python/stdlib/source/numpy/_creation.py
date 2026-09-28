"""Index- and array-construction helpers: mesh/open grids (``mgrid``, ``ogrid``), index tuples
(``ix_``, ``ndindex``, ``unravel_index``, ``ravel_multi_index``), diagonal- and triangle-index
helpers built on ``ndarray.nonzero``, the small array constructors built the same way (``tri``,
``tril``, ``triu``, ``indices``), and the range/diagonal/``*_like`` constructors moved here from
native code (``linspace``, ``logspace``, ``geomspace``, ``eye``, ``identity``, ``diag``,
``meshgrid``, ``zeros_like``, ``ones_like``, ``empty_like``, ``full_like``).

None of these touch storage directly; they compose from ``arange``, ``empty``/``zeros``/``full``,
``reshape``, ``transpose``, and slicing or fancy indexing like any other array code. The
``*_like`` constructors reproduce a prototype's ``order='K'`` memory layout by allocating a
C-contiguous array with the axes permuted into memory order and transposing it back: a pure view
operation, so it costs no more than the allocation itself.
"""

import math

import numpy as np

__all__ = [
    "c_",
    "diag",
    "diag_indices",
    "diag_indices_from",
    "empty_like",
    "eye",
    "fill_diagonal",
    "full_like",
    "geomspace",
    "identity",
    "index_exp",
    "indices",
    "ix_",
    "linspace",
    "logspace",
    "meshgrid",
    "mgrid",
    "ndindex",
    "ogrid",
    "ones_like",
    "r_",
    "ravel_multi_index",
    "s_",
    "tri",
    "tril",
    "tril_indices",
    "tril_indices_from",
    "triu",
    "triu_indices",
    "triu_indices_from",
    "unravel_index",
    "zeros_like",
]


def ix_(*sequences):
    """Index arrays that select the open mesh of `sequences`: ``a[np.ix_(rows, cols)]``."""
    result = []
    ndim = len(sequences)
    for axis, seq in enumerate(sequences):
        seq = np.asanyarray(seq)
        if seq.dtype == np.bool_:
            seq = seq.nonzero()[0]
        shape = [1] * ndim
        shape[axis] = seq.size
        result.append(seq.reshape(shape))
    return tuple(result)


def _unravel_one(flat, shape, order):
    """The coordinate tuple (of plain ints) for one flat position, without array machinery."""
    size = 1
    for dim in shape:
        size *= dim
    if flat < 0 or flat >= max(size, 1):
        raise ValueError(f"index {flat} is out of bounds for array with size {size}")
    axes = range(len(shape) - 1, -1, -1) if order == "C" else range(len(shape))
    coordinates = [0] * len(shape)
    remainder = flat
    for axis in axes:
        dim = shape[axis]
        # Peel the fastest-varying axis off first: its coordinate is what's left over after
        # dividing out that axis's size, and the quotient carries into the slower axes.
        remainder, coordinates[axis] = (0, 0) if dim == 0 else divmod(remainder, dim)
    return tuple(coordinates)


def unravel_index(indices, shape, order="C"):
    """The per-axis coordinates of flat position(s) `indices` into an array of `shape`."""
    if isinstance(indices, (int, np.integer)):
        return _unravel_one(int(indices), shape, order)
    flat_list = np.asanyarray(indices).reshape(-1).tolist()
    per_axis = [[] for _ in shape]
    for flat in flat_list:
        for axis, value in enumerate(_unravel_one(int(flat), shape, order)):
            per_axis[axis].append(value)
    original_shape = np.asanyarray(indices).shape
    return tuple(np.array(values, dtype=np.int64).reshape(original_shape) for values in per_axis)


def ravel_multi_index(multi_index, dims, mode="raise", order="C"):
    """The flat position that `multi_index` (one array per axis) addresses in shape `dims`."""
    if not dims:
        return 0
    scalar = all(np.ndim(c) == 0 for c in multi_index)
    coordinates = [np.asanyarray(c).reshape(-1) for c in multi_index]
    modes = list(mode) if isinstance(mode, (list, tuple)) else [mode] * len(dims)
    count = coordinates[0].size if coordinates else 0
    flat = [0] * count
    axes = range(len(dims) - 1, -1, -1) if order == "C" else range(len(dims))
    for position in range(count):
        stride = 1
        total = 0
        for axis in axes:
            dim = dims[axis]
            value = int(coordinates[axis][position])
            if modes[axis] == "wrap":
                value %= dim
            elif modes[axis] == "clip":
                value = max(0, min(dim - 1, value))
            elif not 0 <= value < dim:
                raise ValueError(f"invalid entry in coordinates array at position {position}")
            total += value * stride
            stride *= dim
        flat[position] = total
    if scalar:
        return flat[0]
    original_shape = np.asanyarray(multi_index[0]).shape
    return np.array(flat, dtype=np.int64).reshape(original_shape)


def ndindex(*shape):
    """Yield every index tuple of an array with `shape`, in C order."""
    if len(shape) == 1 and isinstance(shape[0], tuple):
        shape = shape[0]
    total = 1
    for dim in shape:
        total *= dim
    for flat in range(total):
        yield unravel_index(flat, shape) if shape else ()


class _IndexExpression:
    """Backs ``np.s_``/``np.index_exp``: turns a subscript into the index object it represents."""

    def __init__(self, wrap_single):
        self._wrap_single = wrap_single

    def __getitem__(self, item):
        if isinstance(item, tuple):
            return item
        if self._wrap_single:
            return (item,)
        return item


s_ = _IndexExpression(wrap_single=False)
index_exp = _IndexExpression(wrap_single=True)


def _slice_values(part, grid):
    """The values `part` stands for in ``mgrid``/``ogrid``/``r_``: ``arange(start, stop, step)``,
    or ``count`` evenly spaced points from start to stop inclusive when the step is imaginary
    (``0:1:5j``). The grids space those points as ``start + i * delta`` while ``r_`` uses
    ``linspace``, so the two can differ in the last bit, as they do in NumPy."""
    if not isinstance(part, slice):
        # NumPy reads the step of every grid key, so a non-slice fails on that attribute.
        raise AttributeError(f"'{type(part).__name__}' object has no attribute 'step'")
    start = 0 if part.start is None else part.start
    step = 1 if part.step is None else part.step
    if not isinstance(step, complex):
        return np.arange(start, part.stop, step)
    count = int(abs(step))
    if not grid:
        return np.linspace(start, part.stop, count)
    delta = (part.stop - start) / (count - 1) if count > 1 else 0.0
    return start + np.arange(count) * delta


class _MeshGrid:
    """``np.mgrid``/``np.ogrid``: a dense (`sparse=False`) or open (`sparse=True`) coordinate grid."""

    def __init__(self, sparse):
        self._sparse = sparse

    def __getitem__(self, key):
        if not isinstance(key, tuple):
            return _slice_values(key, grid=True)
        axes = [_slice_values(part, grid=True) for part in key]
        dtype = np.result_type(*axes)
        ndim = len(axes)
        shaped = []
        for axis, values in enumerate(axes):
            shape = [1] * ndim
            shape[axis] = values.size
            shaped.append(values.astype(dtype).reshape(shape))
        if self._sparse:
            return tuple(shaped)
        broadcast_shape = np.broadcast_shapes(*(a.shape for a in shaped))
        return np.stack([np.broadcast_to(a, broadcast_shape) for a in shaped], axis=0)


mgrid = _MeshGrid(sparse=False)
ogrid = _MeshGrid(sparse=True)


def _stack_class(column):
    class _Stacker:
        """``np.r_`` concatenates its items along the first axis, each made at least 1-D.
        ``np.c_`` concatenates along the last axis, each made at least 2-D with a 1-D item
        standing as a column. Slices expand to ranges, or to ``linspace`` counts when the step
        is imaginary."""

        def __getitem__(self, key):
            parts = key if isinstance(key, tuple) else (key,)
            arrays = []
            for part in parts:
                if isinstance(part, slice):
                    item = _slice_values(part, grid=False)
                elif isinstance(part, str):
                    raise NotImplementedError("np.r_/np.c_ string directives are not supported")
                else:
                    item = np.atleast_1d(np.asanyarray(part))
                if column and item.ndim == 1:
                    item = item.reshape(-1, 1)
                arrays.append(item)
            return np.concatenate(arrays, axis=-1 if column else 0)

    return _Stacker()


r_ = _stack_class(column=False)
c_ = _stack_class(column=True)


def diag_indices(n, ndim=2):
    """Indices that address the main diagonal of an `ndim`-dimensional, `n`-per-side array."""
    idx = np.arange(n)
    return tuple(idx for _ in range(ndim))


def diag_indices_from(arr):
    """``diag_indices`` sized to match the equal-length dimensions of `arr`."""
    arr = np.asanyarray(arr)
    if arr.ndim < 2 or any(dim != arr.shape[0] for dim in arr.shape):
        raise ValueError("all dimensions of input must be of equal length")
    return diag_indices(arr.shape[0], arr.ndim)


def tril_indices(n, k=0, m=None):
    """Row and column indices of the lower triangle (at or below diagonal `k`) of an n-by-m array."""
    m = n if m is None else m
    mask = np.tril(np.ones((n, m), dtype=np.bool_), k)
    return mask.nonzero()


def tril_indices_from(arr, k=0):
    arr = np.asanyarray(arr)
    if arr.ndim != 2:
        raise ValueError("input array must be 2-d")
    return tril_indices(arr.shape[0], k=k, m=arr.shape[1])


def triu_indices(n, k=0, m=None):
    """Row and column indices of the upper triangle (at or above diagonal `k`) of an n-by-m array."""
    m = n if m is None else m
    mask = np.triu(np.ones((n, m), dtype=np.bool_), k)
    return mask.nonzero()


def triu_indices_from(arr, k=0):
    arr = np.asanyarray(arr)
    if arr.ndim != 2:
        raise ValueError("input array must be 2-d")
    return triu_indices(arr.shape[0], k=k, m=arr.shape[1])


def fill_diagonal(a, val, wrap=False):
    """Fill the main diagonal of `a` with `val`, writing through any view."""
    if a.ndim < 2:
        raise ValueError("array must be at least 2-d")
    if a.ndim > 2 and any(dim != a.shape[0] for dim in a.shape):
        raise ValueError("All dimensions of input must be of equal length")
    if a.ndim == 2:
        rows, cols = a.shape
        step = cols + 1
        end = cols * rows if (wrap and rows > cols) else cols * min(rows, cols)
    else:
        n = a.shape[0]
        step = sum(n**k for k in range(a.ndim))
        end = n**a.ndim
    a.flat[0:end:step] = val


def tri(N, M=None, k=0, dtype=np.float64):
    """An N-by-M array that is 1 at and below diagonal `k`, 0 elsewhere."""
    M = N if M is None else M
    rows = np.arange(N).reshape(N, 1)
    cols = np.arange(M).reshape(1, M)
    return (cols <= rows + k).astype(dtype)


def tril(m, k=0):
    """`m` with the elements above diagonal `k` zeroed."""
    m = np.asanyarray(m)
    mask = tri(m.shape[-2], m.shape[-1], k=k, dtype=np.bool_)
    return np.where(mask, m, 0)


def triu(m, k=0):
    """`m` with the elements below diagonal `k` zeroed."""
    m = np.asanyarray(m)
    mask = tri(m.shape[-2], m.shape[-1], k=k - 1, dtype=np.bool_)
    return np.where(~mask, m, 0)


def indices(dimensions, dtype=np.int64, sparse=False):
    """The per-axis index arrays of an array with shape `dimensions`."""
    dimensions = tuple(dimensions)
    ndim = len(dimensions)
    axes = [
        np.arange(size, dtype=dtype).reshape([size if i == axis else 1 for i in range(ndim)])
        for axis, size in enumerate(dimensions)
    ]
    if sparse:
        return tuple(axes)
    return np.stack([np.broadcast_to(a, dimensions) for a in axes], axis=0)


def _linspace_values(start, stop, num, endpoint):
    """`(values, step)`: `num` samples at `start + i * step`, NumPy's own `linspace` formula,
    with the last sample pinned to `stop` when `endpoint`. `start`/`stop` are plain floats, one
    real or imaginary part at a time; :func:`linspace` combines two calls for complex bounds."""
    divisions = max(num - 1, 0) if endpoint else num
    delta = stop - start
    if divisions > 0:
        step = delta / divisions
        if step == 0.0:
            values = start + np.arange(num, dtype=np.float64) / divisions * delta
        else:
            values = start + np.arange(num, dtype=np.float64) * step
    else:
        step = float("nan")
        values = np.full(num, start, dtype=np.float64)
    if endpoint and num > 1:
        values[-1] = stop
    return values, step


def _sample_count(num):
    num = int(num)
    if num < 0:
        raise ValueError(f"Number of samples, {num}, must be non-negative.")
    return num


def _scalar_float(value):
    return float(np.asanyarray(value).item())


def _scalar_complex(value):
    return complex(np.asanyarray(value).item())


def linspace(start, stop, num=50, endpoint=True, retstep=False, dtype=None, axis=0, *, device=None):
    """`num` samples evenly spaced from `start` to `stop` (inclusive unless `endpoint=False`)."""
    if axis != 0:
        raise NotImplementedError("linspace() with axis= is not supported")
    num = _sample_count(num)
    if np.iscomplexobj(start) or np.iscomplexobj(stop):
        start_c, stop_c = _scalar_complex(start), _scalar_complex(stop)
        values_re, step_re = _linspace_values(start_c.real, stop_c.real, num, endpoint)
        values_im, step_im = _linspace_values(start_c.imag, stop_c.imag, num, endpoint)
        result_dtype = np.dtype(np.complex128) if dtype is None else np.dtype(dtype)
        if result_dtype.kind != "c":
            raise NotImplementedError(
                "linspace() with complex bounds and a real dtype is not supported"
            )
        result = (values_re + 1j * values_im).astype(result_dtype)
        step = np.complex128(complex(step_re, step_im))
    else:
        values, step_value = _linspace_values(_scalar_float(start), _scalar_float(stop), num, endpoint)
        result_dtype = np.dtype(np.float64) if dtype is None else np.dtype(dtype)
        if result_dtype.kind in "iu":
            values = np.floor(values)
        result = values.astype(result_dtype)
        step = np.float64(step_value)
    return (result, step) if retstep else result


def logspace(start, stop, num=50, endpoint=True, base=10.0, dtype=None, axis=0):
    """`num` samples evenly spaced on a log scale: `base` raised to a :func:`linspace` exponent."""
    if axis != 0:
        raise NotImplementedError("logspace() with axis= is not supported")
    num = _sample_count(num)
    exponents, _ = _linspace_values(_scalar_float(start), _scalar_float(stop), num, endpoint)
    values = np.power(float(base), exponents)
    result_dtype = np.float64 if dtype is None else np.dtype(dtype)
    return values.astype(result_dtype)


def geomspace(start, stop, num=50, endpoint=True, dtype=None, axis=0):
    """`num` samples in geometric progression from `start` to `stop`; both must be nonzero and
    the same sign."""
    if axis != 0:
        raise NotImplementedError("geomspace() with axis= is not supported")
    start_f, stop_f = _scalar_float(start), _scalar_float(stop)
    if start_f == 0.0 or stop_f == 0.0:
        raise ValueError("Geometric sequence cannot include zero")
    if (start_f < 0.0) != (stop_f < 0.0):
        raise NotImplementedError("geomspace() between bounds of different signs is not supported")
    num = _sample_count(num)
    sign = -1.0 if start_f < 0.0 else 1.0
    exponents, _ = _linspace_values(math.log10(abs(start_f)), math.log10(abs(stop_f)), num, endpoint)
    values = sign * np.power(10.0, exponents)
    if values.size:
        values[0] = start_f
    if endpoint and num > 1:
        values[-1] = stop_f
    result_dtype = np.float64 if dtype is None else np.dtype(dtype)
    return values.astype(result_dtype)


def _dimension(value):
    size = int(value)
    if size < 0:
        raise ValueError("negative dimensions are not allowed")
    return size


def eye(N, M=None, k=0, dtype=np.float64, order="C", *, device=None, like=None):
    """An N-by-M matrix that is 1 on diagonal `k`, 0 elsewhere."""
    rows = _dimension(N)
    columns = rows if M is None else _dimension(M)
    k = int(k)
    order = "C" if order is None else order
    rows_idx = np.arange(rows).reshape(rows, 1)
    cols_idx = np.arange(columns).reshape(1, columns)
    base = (cols_idx - rows_idx == k).astype(np.int64).astype(dtype)
    if order in ("F", "f"):
        return np.asfortranarray(base)
    if order not in ("C", "c"):
        raise ValueError(f"order must be one of 'C', 'F', 'A', or 'K' (got {order!r})")
    return base


def identity(n, dtype=np.float64, *, like=None):
    """The `n`-by-`n` identity matrix."""
    return eye(n, n, 0, dtype)


def diag(v, k=0):
    """A matrix with vector `v` on diagonal `k`, or the diagonal `k` of matrix `v`."""
    v = np.asanyarray(v)
    k = int(k)
    if v.ndim == 1:
        length = v.shape[0]
        size = length + abs(k)
        result = np.zeros((size, size), dtype=v.dtype)
        step = size + 1
        start = k if k >= 0 else -k * size
        result.flat[start : start + length * step : step] = v
        return result
    if v.ndim == 2:
        return np.diagonal(v, k)
    raise ValueError("Input must be 1- or 2-d.")


def meshgrid(*xi, indexing="xy", sparse=False, copy=True):
    """Coordinate matrices (or, if `sparse`, open coordinate vectors) from 1-D arrays `xi`."""
    if sparse:
        raise NotImplementedError("meshgrid() with sparse=True is not supported")
    if indexing not in ("xy", "ij"):
        raise ValueError("Valid values for `indexing` are 'xy' and 'ij'.")
    inputs = [np.asanyarray(x).reshape(-1) for x in xi]
    swap = indexing == "xy" and len(inputs) >= 2
    shape = [x.size for x in inputs]
    if swap:
        shape[0], shape[1] = shape[1], shape[0]
    grids = []
    for position, x in enumerate(inputs):
        axis = {0: 1, 1: 0}.get(position, position) if swap else position
        new_shape = [1] * len(inputs)
        new_shape[axis] = x.size
        view = np.broadcast_to(x.reshape(new_shape), shape)
        grids.append(view.copy() if copy else view)
    return tuple(grids)


def _layout_axes(order, prototype, ndim):
    """The axis order (outermost to innermost) NumPy's `*_like` constructors give a new array of
    `ndim` dimensions modeled on `prototype`'s memory layout (`PyArray_NewLikeArrayWithShape`):
    `K` keeps a C- or Fortran-contiguous prototype's order and otherwise sorts axes by decreasing
    absolute stride; it falls back to `C` order when the ranks differ."""
    order = "K" if order is None else order
    if order == "A":
        order = "F" if (prototype.flags.f_contiguous and not prototype.flags.c_contiguous) else "C"
    elif order not in ("C", "F", "K"):
        raise ValueError(f"order must be one of 'C', 'F', 'A', or 'K' (got {order!r})")
    if order == "K":
        if ndim != prototype.ndim or ndim <= 1 or prototype.flags.c_contiguous:
            order = "C"
        elif prototype.flags.f_contiguous:
            order = "F"
        else:
            return tuple(sorted(range(ndim), key=lambda axis: -abs(prototype.strides[axis])))
    if order == "F":
        return tuple(range(ndim - 1, -1, -1))
    return tuple(range(ndim))


def _shape_tuple(value):
    return (int(value),) if isinstance(value, (int, np.integer)) else tuple(int(d) for d in value)


def _like(prototype, dtype, shape, order, build):
    """`build(permuted_shape, dtype)`, C-contiguous, transposed back so its memory order matches
    `prototype`'s under `order`. Allocating the permuted shape directly and transposing (a view)
    costs no more than `build` itself already does."""
    axes = _layout_axes(order, prototype, len(shape))
    permuted = tuple(shape[axis] for axis in axes)
    inverse = [0] * len(axes)
    for position, axis in enumerate(axes):
        inverse[axis] = position
    return build(permuted, dtype).transpose(tuple(inverse))


def zeros_like(a, dtype=None, order="K", subok=True, shape=None, *, device=None):
    a = np.asanyarray(a)
    dtype = a.dtype if dtype is None else np.dtype(dtype)
    shape = a.shape if shape is None else _shape_tuple(shape)
    return _like(a, dtype, shape, order, lambda s, d: np.zeros(s, dtype=d))


def empty_like(prototype, dtype=None, order="K", subok=True, shape=None, *, device=None):
    prototype = np.asanyarray(prototype)
    dtype = prototype.dtype if dtype is None else np.dtype(dtype)
    shape = prototype.shape if shape is None else _shape_tuple(shape)
    return _like(prototype, dtype, shape, order, lambda s, d: np.empty(s, dtype=d))


def ones_like(a, dtype=None, order="K", subok=True, shape=None, *, device=None):
    a = np.asanyarray(a)
    dtype = a.dtype if dtype is None else np.dtype(dtype)
    shape = a.shape if shape is None else _shape_tuple(shape)
    return _like(a, dtype, shape, order, lambda s, d: np.ones(s, dtype=d))


def full_like(a, fill_value, dtype=None, order="K", subok=True, shape=None, *, device=None):
    a = np.asanyarray(a)
    dtype = a.dtype if dtype is None else np.dtype(dtype)
    shape = a.shape if shape is None else _shape_tuple(shape)
    return _like(a, dtype, shape, order, lambda s, d: np.full(s, fill_value, dtype=d))
