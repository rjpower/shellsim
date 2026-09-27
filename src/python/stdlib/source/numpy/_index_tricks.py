"""Index helpers from ``numpy/lib/_index_tricks_impl.py``: ``ndindex``, ``ndenumerate``,
``ix_``, ``index_exp`` and ``s_``.

``ndenumerate`` walks ``ndindex`` over the array's shape, where NumPy reads ``arr.flat.coords``;
both give C-order positions and elements.
"""

import itertools

import numpy as np


class ndindex:
    """An N-dimensional iterator over the index tuples of an array of the given shape."""

    def __init__(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], tuple):
            shape = shape[0]
        if min(shape, default=0) < 0:
            raise ValueError("negative dimensions are not allowed")
        self._iter = itertools.product(*map(range, shape))

    def __iter__(self):
        return self

    def __next__(self):
        return next(self._iter)


class ndenumerate:
    """Pairs of array coordinates and values, in C order."""

    def __init__(self, arr):
        self._arr = np.asarray(arr)
        self._iter = ndindex(self._arr.shape)

    def __iter__(self):
        return self

    def __next__(self):
        index = next(self._iter)
        return index, self._arr[index]


def ix_(*args):
    out = []
    nd = len(args)
    for k, new in enumerate(args):
        if not isinstance(new, np.ndarray):
            new = np.asarray(new)
            if new.size == 0:
                # Explicitly type empty arrays to avoid float default
                new = new.astype(np.intp)
        if new.ndim != 1:
            raise ValueError("Cross index must be 1 dimensional")
        if np.issubdtype(new.dtype, np.bool):
            (new,) = new.nonzero()
        new = new.reshape((1,) * k + (new.size,) + (1,) * (nd - k - 1))
        out.append(new)
    return tuple(out)


class IndexExpression:
    """Build index tuples with indexing syntax: ``np.s_[1:3, ::2]``."""

    def __init__(self, maketuple):
        self.maketuple = maketuple

    def __getitem__(self, item):
        if self.maketuple and not isinstance(item, tuple):
            return (item,)
        return item


index_exp = IndexExpression(maketuple=True)
s_ = IndexExpression(maketuple=False)
