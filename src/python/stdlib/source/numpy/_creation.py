"""Index- and array-construction helpers: mesh/open grids (``mgrid``, ``ogrid``), index tuples
(``ix_``, ``ndindex``, ``unravel_index``, ``ravel_multi_index``), diagonal- and triangle-index
helpers built on ``ndarray.nonzero``, and the small array constructors built the same way
(``tri``, ``tril``, ``triu``, ``indices``).

None of these touch storage directly; they compose from ``arange``, ``reshape``, and the array
constructors like any other array code.
"""

import numpy as np

__all__ = [
    "c_",
    "diag_indices",
    "diag_indices_from",
    "fill_diagonal",
    "index_exp",
    "indices",
    "ix_",
    "mgrid",
    "ndindex",
    "ogrid",
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
