"""Index-construction helpers: ``ix_``, ``ndindex``, ``mgrid``/``ogrid``, ``r_``/``c_``, and the
diagonal- and triangle-index helpers built on ``ndarray.nonzero``.

These build plain index tuples and coordinate arrays; none of them touch storage directly, so
they compose from ``arange``, ``reshape``, and the array constructors like any other array code.
"""

import numpy as np

__all__ = [
    "c_",
    "diag_indices",
    "diag_indices_from",
    "fill_diagonal",
    "index_exp",
    "ix_",
    "mgrid",
    "ndindex",
    "ogrid",
    "r_",
    "ravel_multi_index",
    "s_",
    "tril_indices",
    "tril_indices_from",
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


def _grid_axis(part, want_complex_as_count):
    if isinstance(part, slice):
        start = 0 if part.start is None else part.start
        stop = part.stop
        step = 1 if part.step is None else part.step
        if want_complex_as_count and isinstance(step, complex):
            count = int(abs(step))
            return np.linspace(start, stop, count)
        return np.arange(start, stop, step)
    raise TypeError("mgrid/ogrid indices must be slices")


class _MeshGrid:
    """``np.mgrid``/``np.ogrid``: a dense (`sparse=False`) or open (`sparse=True`) coordinate grid."""

    def __init__(self, sparse):
        self._sparse = sparse

    def __getitem__(self, key):
        parts = key if isinstance(key, tuple) else (key,)
        axes = [_grid_axis(part, want_complex_as_count=True) for part in parts]
        ndim = len(axes)
        shaped = []
        for axis, values in enumerate(axes):
            shape = [1] * ndim
            shape[axis] = values.size
            shaped.append(values.reshape(shape))
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
                    item = _grid_axis(part, want_complex_as_count=True)
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
