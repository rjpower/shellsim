"""Set operations on arrays, ported from ``numpy/lib/_arraysetops_impl.py`` (NumPy 2.5).

The functions keep NumPy's algorithms: ``unique`` sorts and masks runs of equal values, and
``isin`` chooses the lookup-table, pairwise, or sorting method by NumPy's size rules. Two
parts differ:

- ``unique(..., axis=k)`` sorts rows with ``lexsort`` instead of viewing each row as one
  structured element. Structured dtypes compare fields in order, so the result is the same.
- NumPy returns the ``unique_all``, ``unique_counts`` and ``unique_inverse`` results as named
  tuples. shellsim has no ``namedtuple``, so they are small classes that unpack, index and
  name their fields in the same way.
"""

import numpy as np

__all__ = [
    "ediff1d",
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


def ediff1d(ary, to_end=None, to_begin=None):
    ary = np.asanyarray(ary).ravel()
    dtype_req = ary.dtype
    if to_begin is None and to_end is None:
        return ary[1:] - ary[:-1]
    if to_begin is None:
        l_begin = 0
    else:
        to_begin = np.asanyarray(to_begin)
        if not np.can_cast(to_begin, dtype_req, casting="same_kind"):
            raise TypeError(
                "dtype of `to_begin` must be compatible with input `ary` under the "
                "`same_kind` rule."
            )
        to_begin = to_begin.ravel()
        l_begin = len(to_begin)
    if to_end is None:
        l_end = 0
    else:
        to_end = np.asanyarray(to_end)
        if not np.can_cast(to_end, dtype_req, casting="same_kind"):
            raise TypeError(
                "dtype of `to_end` must be compatible with input `ary` under the "
                "`same_kind` rule."
            )
        to_end = to_end.ravel()
        l_end = len(to_end)
    l_diff = max(len(ary) - 1, 0)
    result = np.empty_like(ary, shape=l_diff + l_begin + l_end)
    if l_begin > 0:
        result[:l_begin] = to_begin
    if l_end > 0:
        result[l_begin + l_diff :] = to_end
    np.subtract(ary[1:], ary[:-1], result[l_begin : l_begin + l_diff])
    return result


def _unpack_tuple(x):
    if len(x) == 1:
        return x[0]
    return x


def unique(
    ar,
    return_index=False,
    return_inverse=False,
    return_counts=False,
    axis=None,
    *,
    equal_nan=True,
    sorted=True,
):
    ar = np.asanyarray(ar)
    if axis is None or ar.ndim == 1:
        if axis is not None:
            np._normalize_axis_index(axis, ar.ndim)
        ret = _unique1d(
            ar,
            return_index,
            return_inverse,
            return_counts,
            equal_nan=equal_nan,
            inverse_shape=ar.shape,
            axis=None,
        )
        return _unpack_tuple(ret)

    try:
        ar = np.moveaxis(ar, axis, 0)
    except np.exceptions.AxisError:
        # This removes the "source" prefix from the error message.
        raise np.exceptions.AxisError(axis, ar.ndim) from None

    orig_shape, orig_dtype = ar.shape, ar.dtype
    ar = ar.reshape(orig_shape[0], np.prod(orig_shape[1:], dtype=np.intp))
    ar = np.ascontiguousarray(ar)
    if ar.dtype == object:
        raise TypeError(f"The axis argument to unique is not supported for dtype {ar.dtype}")

    def reshape_uniq(uniq):
        n = len(uniq)
        uniq = uniq.reshape(n, *orig_shape[1:])
        return np.moveaxis(uniq, 0, axis)

    output = _unique_rows(ar, return_index, return_inverse, return_counts)
    output = (reshape_uniq(output[0]),) + output[1:]
    return _unpack_tuple(output)


def _unique_rows(ar, return_index, return_inverse, return_counts):
    """``_unique1d`` for the rows of a 2-d array, compared lexicographically."""
    if ar.shape[1] > 0:
        perm = np.lexsort(ar.T[::-1])
    else:
        perm = np.arange(ar.shape[0])
    aux = ar[perm]
    mask = np.empty(aux.shape[0], dtype=np.bool)
    mask[:1] = True
    mask[1:] = np.any(aux[1:] != aux[:-1], axis=1)
    ret = (aux[mask],)
    if return_index:
        ret += (perm[mask],)
    if return_inverse:
        imask = np.cumsum(mask) - 1
        inv_idx = np.empty(mask.shape, dtype=np.intp)
        inv_idx[perm] = imask
        ret += (inv_idx,)
    if return_counts:
        idx = np.concatenate(np.nonzero(mask) + ([mask.size],))
        ret += (np.diff(idx),)
    return ret


def _unique1d(
    ar,
    return_index=False,
    return_inverse=False,
    return_counts=False,
    *,
    equal_nan=True,
    inverse_shape=None,
    axis=None,
):
    """Find the unique elements of an array, ignoring shape.

    NumPy uses a hash table when no indices or counts are requested and sorts the result when
    ``sorted=True``; sorting throughout gives the same values in the same order.
    """
    ar = np.asanyarray(ar).flatten()

    optional_indices = return_index or return_inverse

    if optional_indices:
        perm = ar.argsort(kind="mergesort" if return_index else "quicksort")
        aux = ar[perm]
    else:
        ar.sort()
        aux = ar
    mask = np.empty(aux.shape, dtype=np.bool)
    mask[:1] = True
    if equal_nan and aux.shape[0] > 0 and aux.dtype.kind in "cfmM" and np.isnan(aux[-1]):
        if aux.dtype.kind == "c":  # for complex all NaNs are considered equivalent
            aux_firstnan = np.searchsorted(np.isnan(aux), True, side="left")
        else:
            aux_firstnan = np.searchsorted(aux, aux[-1], side="left")
        if aux_firstnan > 0:
            mask[1:aux_firstnan] = aux[1:aux_firstnan] != aux[: aux_firstnan - 1]
        mask[aux_firstnan] = True
        mask[aux_firstnan + 1 :] = False
    else:
        mask[1:] = aux[1:] != aux[:-1]

    ret = (aux[mask],)
    if return_index:
        ret += (perm[mask],)
    if return_inverse:
        imask = np.cumsum(mask) - 1
        inv_idx = np.empty(mask.shape, dtype=np.intp)
        inv_idx[perm] = imask
        ret += (inv_idx.reshape(inverse_shape) if axis is None else inv_idx,)
    if return_counts:
        idx = np.concatenate(np.nonzero(mask) + ([mask.size],))
        ret += (np.diff(idx),)
    return ret


class _Result:
    """A fixed set of named results that unpacks and indexes like NumPy's named tuples."""

    _fields = ()

    def __init__(self, *values):
        self._values = values
        for name, value in zip(self._fields, values):
            setattr(self, name, value)

    def __len__(self):
        return len(self._values)

    def __iter__(self):
        return iter(self._values)

    def __getitem__(self, index):
        return self._values[index]

    def __eq__(self, other):
        return tuple(self) == tuple(other)

    def __repr__(self):
        fields = ", ".join(
            f"{name}={value!r}" for name, value in zip(self._fields, self._values)
        )
        return f"{type(self).__name__}({fields})"


class UniqueAllResult(_Result):
    _fields = ("values", "indices", "inverse_indices", "counts")


class UniqueCountsResult(_Result):
    _fields = ("values", "counts")


class UniqueInverseResult(_Result):
    _fields = ("values", "inverse_indices")


def unique_all(x):
    result = unique(
        x, return_index=True, return_inverse=True, return_counts=True, equal_nan=False
    )
    return UniqueAllResult(*result)


def unique_counts(x):
    result = unique(
        x, return_index=False, return_inverse=False, return_counts=True, equal_nan=False
    )
    return UniqueCountsResult(*result)


def unique_inverse(x):
    result = unique(
        x, return_index=False, return_inverse=True, return_counts=False, equal_nan=False
    )
    return UniqueInverseResult(*result)


def unique_values(x):
    return unique(
        x,
        return_index=False,
        return_inverse=False,
        return_counts=False,
        equal_nan=False,
        sorted=False,
    )


def intersect1d(ar1, ar2, assume_unique=False, return_indices=False):
    ar1 = np.asanyarray(ar1)
    ar2 = np.asanyarray(ar2)

    if not assume_unique:
        if return_indices:
            ar1, ind1 = unique(ar1, return_index=True)
            ar2, ind2 = unique(ar2, return_index=True)
        else:
            ar1 = unique(ar1)
            ar2 = unique(ar2)
    else:
        ar1 = ar1.ravel()
        ar2 = ar2.ravel()

    aux = np.concatenate((ar1, ar2))
    if return_indices:
        aux_sort_indices = np.argsort(aux, kind="mergesort")
        aux = aux[aux_sort_indices]
    else:
        aux.sort()

    mask = aux[1:] == aux[:-1]
    int1d = aux[:-1][mask]

    if return_indices:
        ar1_indices = aux_sort_indices[:-1][mask]
        ar2_indices = aux_sort_indices[1:][mask] - ar1.size
        if not assume_unique:
            ar1_indices = ind1[ar1_indices]
            ar2_indices = ind2[ar2_indices]

        return int1d, ar1_indices, ar2_indices
    return int1d


def setxor1d(ar1, ar2, assume_unique=False):
    if not assume_unique:
        ar1 = unique(ar1)
        ar2 = unique(ar2)

    aux = np.concatenate((ar1, ar2), axis=None)
    if aux.size == 0:
        return aux

    aux.sort()
    flag = np.concatenate(([True], aux[1:] != aux[:-1], [True]))
    return aux[flag[1:] & flag[:-1]]


def _isin(ar1, ar2, assume_unique=False, invert=False, *, kind=None):
    # Ravel both arrays, behavior for the first array could be different
    ar1 = np.asarray(ar1).ravel()
    ar2 = np.asarray(ar2).ravel()

    # Ensure that iteration through object arrays yields size-1 arrays
    if ar2.dtype == object:
        ar2 = ar2.reshape(-1, 1)

    if kind not in {None, "sort", "table"}:
        raise ValueError(f"Invalid kind: '{kind}'. Please use None, 'sort' or 'table'.")

    # Can use the table method if all arrays are integers or boolean:
    is_int_arrays = all(ar.dtype.kind in ("u", "i", "b") for ar in (ar1, ar2))
    use_table_method = is_int_arrays and kind in {None, "table"}

    if use_table_method:
        if ar2.size == 0:
            if invert:
                return np.ones_like(ar1, dtype=bool)
            return np.zeros_like(ar1, dtype=bool)

        # Convert booleans to uint8 so we can use the fast integer algorithm
        if ar1.dtype == bool:
            ar1 = ar1.astype(np.uint8)
        if ar2.dtype == bool:
            ar2 = ar2.astype(np.uint8)

        ar2_min = int(np.min(ar2))
        ar2_max = int(np.max(ar2))

        ar2_range = ar2_max - ar2_min

        # Constraints on whether we can actually use the table method:
        #  1. Assert memory usage is not too large
        below_memory_constraint = ar2_range <= 6 * (ar1.size + ar2.size)
        #  2. Check overflows for (ar2 - ar2_min); dtype=ar2.dtype
        range_safe_from_overflow = ar2_range <= np.iinfo(ar2.dtype).max

        if range_safe_from_overflow and (below_memory_constraint or kind == "table"):
            if invert:
                outgoing_array = np.ones_like(ar1, dtype=bool)
            else:
                outgoing_array = np.zeros_like(ar1, dtype=bool)

            # Make elements 1 where the integer exists in ar2
            if invert:
                isin_helper_ar = np.ones(ar2_range + 1, dtype=bool)
                isin_helper_ar[ar2 - ar2_min] = 0
            else:
                isin_helper_ar = np.zeros(ar2_range + 1, dtype=bool)
                isin_helper_ar[ar2 - ar2_min] = 1

            # Mask out elements we know won't work
            basic_mask = (ar1 <= ar2_max) & (ar1 >= ar2_min)
            in_range_ar1 = ar1[basic_mask]
            if in_range_ar1.size == 0:
                # Nothing more to do, since all values are out of range.
                return outgoing_array

            # Unfortunately, ar2_min can be out of range for `intp` even
            # if the calculation result must fit in range (and be positive).
            # In that case, use ar2.dtype which must work for all unmasked
            # values.
            try:
                ar2_min = np.array(ar2_min, dtype=np.intp)
                dtype = np.intp
            except OverflowError:
                dtype = ar2.dtype

            out = np.empty_like(in_range_ar1, dtype=np.intp)
            outgoing_array[basic_mask] = isin_helper_ar[
                np.subtract(in_range_ar1, ar2_min, dtype=dtype, out=out, casting="unsafe")
            ]

            return outgoing_array
        elif kind == "table":  # not range_safe_from_overflow
            raise RuntimeError(
                "You have specified kind='table', "
                "but the range of values in `ar2` or `ar1` exceed the "
                "maximum integer of the datatype. "
                "Please set `kind` to None or 'sort'."
            )
    elif kind == "table":
        raise ValueError(
            "The 'table' method is only "
            "supported for boolean or integer arrays. "
            "Please select 'sort' or None for kind."
        )

    # Check if one of the arrays may contain arbitrary objects
    contains_object = ar1.dtype.hasobject or ar2.dtype.hasobject

    # This code is run when
    # a) the first condition is true, making the code significantly faster
    # b) the second condition is true (i.e. `ar1` or `ar2` may contain
    #    arbitrary objects), since then sorting is not guaranteed to work
    if len(ar2) < 10 * len(ar1) ** 0.145 or contains_object:
        if invert:
            mask = np.ones(len(ar1), dtype=bool)
            for a in ar2:
                mask &= ar1 != a
        else:
            mask = np.zeros(len(ar1), dtype=bool)
            for a in ar2:
                mask |= ar1 == a
        return mask

    # Otherwise use sorting
    if not assume_unique:
        ar1, rev_idx = unique(ar1, return_inverse=True)
        ar2 = unique(ar2)

    ar = np.concatenate((ar1, ar2))
    # We need this to be a stable sort, so always use 'mergesort'
    # here. The values from the first array should always come before
    # the values from the second array.
    order = ar.argsort(kind="mergesort")
    sar = ar[order]
    if invert:
        bool_ar = sar[1:] != sar[:-1]
    else:
        bool_ar = sar[1:] == sar[:-1]
    flag = np.concatenate((bool_ar, [invert]))
    ret = np.empty(ar.shape, dtype=bool)
    ret[order] = flag

    if assume_unique:
        return ret[: len(ar1)]
    return ret[rev_idx]


def isin(element, test_elements, assume_unique=False, invert=False, *, kind=None):
    element = np.asarray(element)
    return _isin(
        element, test_elements, assume_unique=assume_unique, invert=invert, kind=kind
    ).reshape(element.shape)


def union1d(ar1, ar2):
    return unique(np.concatenate((ar1, ar2), axis=None))


def setdiff1d(ar1, ar2, assume_unique=False):
    if assume_unique:
        ar1 = np.asarray(ar1).ravel()
    else:
        ar1 = unique(ar1)
        ar2 = unique(ar2)
    return ar1[_isin(ar1, ar2, assume_unique=True, invert=True)]
