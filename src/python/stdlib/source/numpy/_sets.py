"""Set operations over array values: ``unique`` and the sorted-array set algebra built on it.

``unique`` sorts with the native, always-stable ``argsort`` and then scans for adjacent equal
runs; every other function here (``isin``, ``intersect1d``, ``union1d``, ``setdiff1d``,
``setxor1d``) is a small composition of ``unique``, broadcasted comparison, and ``searchsorted``.
"""

from collections import namedtuple

import numpy as np

__all__ = [
    "in1d",
    "intersect1d",
    "isin",
    "setdiff1d",
    "setxor1d",
    "union1d",
    "unique",
    "unique_all",
    "unique_counts",
    "unique_inverse",
    "unique_values",
]


UniqueAllResult = namedtuple("UniqueAllResult", ["values", "indices", "inverse_indices", "counts"])
UniqueCountsResult = namedtuple("UniqueCountsResult", ["values", "counts"])
UniqueInverseResult = namedtuple("UniqueInverseResult", ["values", "inverse_indices"])


def _unique_1d(values, equal_nan=True):
    """Sorted unique values of the 1-d array `values`, plus the sort order and a first-seen mask."""
    order = np.argsort(values)
    sorted_values = values[order]
    n = sorted_values.size
    if n == 0:
        mask = np.zeros(0, dtype=np.bool_)
    else:
        if equal_nan and np.issubdtype(sorted_values.dtype, np.floating):
            same = (sorted_values[1:] == sorted_values[:-1]) | (
                np.isnan(sorted_values[1:]) & np.isnan(sorted_values[:-1])
            )
        else:
            same = sorted_values[1:] == sorted_values[:-1]
        mask = np.concatenate([np.ones(1, dtype=np.bool_), ~same])
    return sorted_values[mask], order, mask


def unique(ar, return_index=False, return_inverse=False, return_counts=False, axis=None, *, equal_nan=True):
    """Sorted unique values of `ar`, optionally with first indices, an inverse map, and counts."""
    ar = np.asanyarray(ar)
    original_shape = ar.shape
    if axis is None:
        flat = ar.reshape(-1)
    else:
        axis = int(axis)
        if axis < 0:
            axis += ar.ndim
        moved = np.moveaxis(ar, axis, 0)
        flattened = moved.reshape(moved.shape[0], -1)
        rows = [tuple(row.tolist()) for row in flattened]
        seen = {}
        for position, row in enumerate(rows):
            seen.setdefault(row, []).append(position)
        ordered_rows = sorted(seen)
        values = np.stack([moved[seen[row][0]] for row in ordered_rows], axis=0)
        values = np.moveaxis(values, 0, axis)
        outputs = [values]
        if return_index:
            outputs.append(np.array([seen[row][0] for row in ordered_rows], dtype=np.int64))
        if return_inverse:
            rank = {row: i for i, row in enumerate(ordered_rows)}
            outputs.append(np.array([rank[row] for row in rows], dtype=np.int64))
        if return_counts:
            outputs.append(np.array([len(seen[row]) for row in ordered_rows], dtype=np.int64))
        return outputs[0] if len(outputs) == 1 else tuple(outputs)

    unique_values, order, mask = _unique_1d(flat, equal_nan=equal_nan)
    outputs = [unique_values]
    if return_index:
        outputs.append(order[mask])
    if return_inverse or return_counts:
        rank = np.cumsum(mask) - 1
    if return_inverse:
        inverse = np.empty(flat.size, dtype=np.int64)
        inverse[order] = rank
        outputs.append(inverse.reshape(original_shape))
    if return_counts:
        outputs.append(np.bincount(rank, minlength=unique_values.size))
    return outputs[0] if len(outputs) == 1 else tuple(outputs)


def unique_values(x):
    """The sorted unique values of `x`, flattened; the plain-array form of the Array API result."""
    return unique(x)


def unique_all(x):
    values, indices, inverse, counts = unique(
        x, return_index=True, return_inverse=True, return_counts=True
    )
    return UniqueAllResult(values, indices, inverse.reshape(-1), counts)


def unique_counts(x):
    values, counts = unique(x, return_counts=True)
    return UniqueCountsResult(values, counts)


def unique_inverse(x):
    values, inverse = unique(x, return_inverse=True)
    return UniqueInverseResult(values, inverse)


def isin(element, test_elements, assume_unique=False, invert=False, *, kind=None):
    """A boolean array shaped like `element`: whether each value occurs in `test_elements`."""
    element = np.asanyarray(element)
    test_elements = np.asanyarray(test_elements).reshape(-1)
    if test_elements.size == 0:
        result = np.zeros(element.shape, dtype=np.bool_)
        return ~result if invert else result
    matches = element.reshape(element.shape + (1,)) == test_elements
    result = np.any(matches, axis=-1)
    return ~result if invert else result


in1d = isin


def union1d(ar1, ar2):
    """The sorted union of the unique values of `ar1` and `ar2`."""
    return unique(np.concatenate([np.asanyarray(ar1).reshape(-1), np.asanyarray(ar2).reshape(-1)]))


def intersect1d(ar1, ar2, assume_unique=False, return_indices=False):
    """The sorted values common to `ar1` and `ar2`, each already reduced to its unique values."""
    if assume_unique:
        ar1 = np.asanyarray(ar1).reshape(-1)
        ar2 = np.asanyarray(ar2).reshape(-1)
    else:
        ar1 = unique(ar1)
        ar2 = unique(ar2)
    mask1 = isin(ar1, ar2, assume_unique=True)
    result = ar1[mask1]
    if not return_indices:
        return result
    index1 = np.flatnonzero(mask1)
    index2 = np.searchsorted(ar2, result)
    return result, index1, index2


def setdiff1d(ar1, ar2, assume_unique=False):
    """The sorted, unique values in `ar1` that are not in `ar2`."""
    ar1 = np.asanyarray(ar1).reshape(-1) if assume_unique else unique(ar1)
    ar2 = np.asanyarray(ar2).reshape(-1)
    return ar1[~isin(ar1, ar2, assume_unique=True)]


def setxor1d(ar1, ar2, assume_unique=False):
    """The sorted values that occur in exactly one of `ar1`, `ar2`."""
    if not assume_unique:
        ar1 = unique(ar1)
        ar2 = unique(ar2)
    else:
        ar1 = np.asanyarray(ar1).reshape(-1)
        ar2 = np.asanyarray(ar2).reshape(-1)
    aux = np.sort(np.concatenate([ar1, ar2]))
    if aux.size == 0:
        return aux
    edge = np.ones(1, dtype=np.bool_)
    flag = np.concatenate([edge, aux[1:] != aux[:-1], edge])
    return aux[flag[1:] & flag[:-1]]
