"""Statistics written in Python: ``median``, ``percentile``, ``quantile``, ``cov`` and
``corrcoef``.

The code follows ``numpy/lib/_function_base_impl.py`` step for step, including the thirteen
quantile methods, the weighted ``inverted_cdf`` path, and the NaN propagation that reads the
last element after a partition. ``partition`` here is a full stable sort, which selects the same
order statistics as NumPy's introselect.
"""

import warnings

import numpy as np
from numpy._shape_base import normalize_axis_tuple


def _weights_are_valid(weights, a, axis):
    wgt = np.asanyarray(weights)
    if a.shape != wgt.shape:
        if axis is None:
            raise TypeError("Axis must be specified when shapes of a and weights differ.")
        if wgt.shape != tuple(a.shape[ax] for ax in axis):
            raise ValueError(
                "Shape of weights must be consistent with shape of a along specified axis."
            )
        # set up wgt to broadcast along axis
        wgt = wgt.transpose(np.argsort(axis))
        wgt = wgt.reshape(tuple((s if ax in axis else 1) for ax, s in enumerate(a.shape)))
    return wgt


def _median_nancheck(data, result, axis):
    """Replace the median with NaN in slices whose sorted data ends in NaN."""
    if data.size == 0:
        return result
    potential_nans = data.take(-1, axis=axis)
    n = np.isnan(potential_nans)
    if not n.any():
        return result
    # A scalar result cannot be written in place.
    if isinstance(result, np.generic):
        return potential_nans
    np.copyto(result, potential_nans, where=n)
    return result


def _ureduce(a, func, keepdims=False, **kwargs):
    """Reduce over several axes by moving them to the end and merging them into one."""
    a = np.asanyarray(a)
    axis = kwargs.get("axis")
    out = kwargs.get("out")
    nd = a.ndim
    if axis is not None:
        axis = normalize_axis_tuple(axis, nd)
        if keepdims and out is not None:
            index_out = tuple(0 if i in axis else slice(None) for i in range(nd))
            kwargs["out"] = out[(Ellipsis,) + index_out]
        if len(axis) == 1:
            kwargs["axis"] = axis[0]
        else:
            keep = sorted(set(range(nd)) - set(axis))
            nkeep = len(keep)

            def reshape_arr(a):
                a = np.moveaxis(a, keep, range(nkeep))
                return a.reshape(a.shape[:nkeep] + (-1,))

            a = reshape_arr(a)
            weights = kwargs.get("weights")
            if weights is not None:
                kwargs["weights"] = reshape_arr(weights)
            kwargs["axis"] = -1
    elif keepdims and out is not None:
        index_out = (0,) * nd
        kwargs["out"] = out[(Ellipsis,) + index_out]
    r = func(a, **kwargs)
    if out is not None:
        return out
    if keepdims:
        if axis is None:
            index_r = (np.newaxis,) * nd
        else:
            index_r = tuple(np.newaxis if i in axis else slice(None) for i in range(nd))
        r = r[(Ellipsis,) + index_r]
    return r


def median(a, axis=None, out=None, overwrite_input=False, keepdims=False):
    return _ureduce(
        a, func=_median, keepdims=keepdims, axis=axis, out=out, overwrite_input=overwrite_input
    )


def _median(a, axis=None, out=None, overwrite_input=False):
    a = np.asanyarray(a)
    if axis is None:
        sz = a.size
    else:
        sz = a.shape[axis]
    if sz % 2 == 0:
        szh = sz // 2
        kth = [szh - 1, szh]
    else:
        kth = [(sz - 1) // 2]
    # A NaN sorts last, so the last element tells whether the slice has one.
    supports_nans = np.issubdtype(a.dtype, np.inexact) or a.dtype.kind in "Mm"
    if supports_nans:
        kth.append(-1)
    if overwrite_input:
        if axis is None:
            part = a.ravel()
            part.partition(kth)
        else:
            a.partition(kth, axis=axis)
            part = a
    else:
        part = np.partition(a, kth, axis=axis)
    if part.shape == ():
        return part.item()
    if axis is None:
        axis = 0
    indexer = [slice(None)] * part.ndim
    index = part.shape[axis] // 2
    if part.shape[axis] % 2 == 1:
        indexer[axis] = slice(index, index + 1)
    else:
        indexer[axis] = slice(index - 1, index + 1)
    indexer = tuple(indexer)
    rout = np.mean(part[indexer], axis=axis, out=out)
    if supports_nans and sz > 0:
        rout = _median_nancheck(part, rout, axis)
    return rout


def _check_weights(weights, a, axis, method):
    if method != "inverted_cdf":
        raise ValueError(f"Only method 'inverted_cdf' supports weights. Got: {method}.")
    if axis is not None:
        axis = normalize_axis_tuple(axis, a.ndim, argname="axis")
    weights = _weights_are_valid(weights=weights, a=a, axis=axis)
    if np.any(weights < 0):
        raise ValueError("Weights must be non-negative.")
    return weights


def percentile(
    a,
    q,
    axis=None,
    out=None,
    overwrite_input=False,
    method="linear",
    keepdims=False,
    *,
    weights=None,
):
    a = np.asanyarray(a)
    if a.dtype.kind == "c":
        raise TypeError("a must be an array of real numbers")
    weak_q = type(q) in (int, float)
    q = np.true_divide(q, 100, out=...)
    if not _quantile_is_valid(q):
        raise ValueError("Percentiles must be in the range [0, 100]")
    if weights is not None:
        weights = _check_weights(weights, a, axis, method)
    return _quantile_unchecked(
        a, q, axis, out, overwrite_input, method, keepdims, weights, weak_q
    )


def quantile(
    a,
    q,
    axis=None,
    out=None,
    overwrite_input=False,
    method="linear",
    keepdims=False,
    *,
    weights=None,
):
    a = np.asanyarray(a)
    if a.dtype.kind == "c":
        raise TypeError("a must be an array of real numbers")
    weak_q = type(q) in (int, float)
    q = np.asanyarray(q)
    if not _quantile_is_valid(q):
        raise ValueError("Quantiles must be in the range [0, 1]")
    if weights is not None:
        weights = _check_weights(weights, a, axis, method)
    return _quantile_unchecked(
        a, q, axis, out, overwrite_input, method, keepdims, weights, weak_q
    )


def _quantile_unchecked(
    a,
    q,
    axis=None,
    out=None,
    overwrite_input=False,
    method="linear",
    keepdims=False,
    weights=None,
    weak_q=False,
):
    return _ureduce(
        a,
        func=_quantile_ureduce_func,
        q=q,
        weights=weights,
        keepdims=keepdims,
        axis=axis,
        out=out,
        overwrite_input=overwrite_input,
        method=method,
        weak_q=weak_q,
    )


def _quantile_is_valid(q):
    # avoid expensive reductions, relevant for arrays with < O(1000) elements
    if q.ndim == 1 and q.size < 10:
        for i in range(q.size):
            if not (0.0 <= q[i] <= 1.0):
                return False
    elif not (q.min() >= 0 and q.max() <= 1):
        return False
    return True


def _compute_virtual_index(n, quantiles, alpha, beta):
    return n * quantiles + (alpha + quantiles * (1 - alpha - beta)) - 1


def _get_gamma(virtual_indexes, previous_indexes, method):
    gamma = np.asanyarray(virtual_indexes - previous_indexes)
    gamma = method["fix_gamma"](gamma, virtual_indexes)
    return np.asanyarray(gamma, dtype=virtual_indexes.dtype)


def _lerp(a, b, t, out=None):
    """Interpolate from ``a`` toward ``b``, computing from ``b`` for ``t >= 0.5`` for accuracy."""
    diff_b_a = b - a
    lerp_interpolation = np.add(a, diff_b_a * t, out=... if out is None else out)
    np.subtract(
        b,
        diff_b_a * (1 - t),
        out=lerp_interpolation,
        where=t >= 0.5,
        casting="unsafe",
        dtype=lerp_interpolation.dtype,
    )
    if lerp_interpolation.ndim == 0 and out is None:
        lerp_interpolation = lerp_interpolation[()]
    return lerp_interpolation


def _get_gamma_mask(shape, default_value, conditioned_value, where):
    out = np.full(shape, default_value)
    np.copyto(out, conditioned_value, where=where, casting="unsafe")
    return out


def _discrete_interpolation_to_boundaries(index, gamma_condition_fun):
    previous = np.floor(index)
    next = previous + 1
    gamma = index - previous
    res = _get_gamma_mask(
        shape=index.shape,
        default_value=next,
        conditioned_value=previous,
        where=gamma_condition_fun(gamma, index),
    ).astype(np.intp)
    # Some methods can lead to out-of-bound integers, clip them:
    res[res < 0] = 0
    return res


def _closest_observation(n, quantiles):
    # "choose the nearest even order statistic at g=0" (H&F (1996) pp. 362).
    def gamma_fun(gamma, index):
        return (gamma == 0) & (np.floor(index) % 2 == 1)

    return _discrete_interpolation_to_boundaries(n * quantiles - 1 - 0.5, gamma_fun)


def _inverted_cdf(n, quantiles):
    def gamma_fun(gamma, _):
        return gamma == 0

    return _discrete_interpolation_to_boundaries(n * quantiles - 1, gamma_fun)


def _identity_gamma(gamma, _):
    return gamma


_QuantileMethods = {
    # --- HYNDMAN and FAN METHODS
    # Discrete methods
    "inverted_cdf": {
        "get_virtual_index": lambda n, quantiles: _inverted_cdf(n, quantiles),
        "fix_gamma": None,
    },
    "averaged_inverted_cdf": {
        "get_virtual_index": lambda n, quantiles: n * quantiles - 1,
        "fix_gamma": lambda gamma, _: _get_gamma_mask(
            shape=gamma.shape, default_value=1.0, conditioned_value=0.5, where=gamma == 0
        ),
    },
    "closest_observation": {
        "get_virtual_index": lambda n, quantiles: _closest_observation(n, quantiles),
        "fix_gamma": None,
    },
    # Continuous methods
    "interpolated_inverted_cdf": {
        "get_virtual_index": lambda n, quantiles: _compute_virtual_index(n, quantiles, 0, 1),
        "fix_gamma": _identity_gamma,
    },
    "hazen": {
        "get_virtual_index": lambda n, quantiles: _compute_virtual_index(n, quantiles, 0.5, 0.5),
        "fix_gamma": _identity_gamma,
    },
    "weibull": {
        "get_virtual_index": lambda n, quantiles: _compute_virtual_index(n, quantiles, 0, 0),
        "fix_gamma": _identity_gamma,
    },
    # Default method.
    "linear": {
        "get_virtual_index": lambda n, quantiles: (n - 1) * quantiles,
        "fix_gamma": _identity_gamma,
    },
    "median_unbiased": {
        "get_virtual_index": lambda n, quantiles: _compute_virtual_index(
            n, quantiles, 1 / 3.0, 1 / 3.0
        ),
        "fix_gamma": _identity_gamma,
    },
    "normal_unbiased": {
        "get_virtual_index": lambda n, quantiles: _compute_virtual_index(
            n, quantiles, 3 / 8.0, 3 / 8.0
        ),
        "fix_gamma": _identity_gamma,
    },
    # --- OTHER METHODS
    "lower": {
        "get_virtual_index": lambda n, quantiles: np.floor((n - 1) * quantiles).astype(np.intp),
        "fix_gamma": None,
    },
    "higher": {
        "get_virtual_index": lambda n, quantiles: np.ceil((n - 1) * quantiles).astype(np.intp),
        "fix_gamma": None,
    },
    "midpoint": {
        "get_virtual_index": lambda n, quantiles: 0.5
        * (np.floor((n - 1) * quantiles) + np.ceil((n - 1) * quantiles)),
        "fix_gamma": lambda gamma, index: _get_gamma_mask(
            shape=gamma.shape, default_value=0.5, conditioned_value=0.0, where=index % 1 == 0
        ),
    },
    "nearest": {
        "get_virtual_index": lambda n, quantiles: np.around((n - 1) * quantiles).astype(np.intp),
        "fix_gamma": None,
    },
}


def _quantile_ureduce_func(
    a, q, weights, axis=None, out=None, overwrite_input=False, method="linear", weak_q=False
):
    if q.ndim > 2:
        # The code below works fine for nd, but it might not have useful
        # semantics. For now, keep the supported dimensions the same as it was
        # before.
        raise ValueError("q must be a scalar or 1d")
    if overwrite_input:
        if axis is None:
            axis = 0
            arr = a.ravel()
            wgt = None if weights is None else weights.ravel()
        else:
            arr = a
            wgt = weights
    elif axis is None:
        axis = 0
        arr = a.flatten()
        wgt = None if weights is None else weights.flatten()
    else:
        arr = a.copy()
        wgt = weights
    return _quantile(
        arr, quantiles=q, axis=axis, method=method, out=out, weights=wgt, weak_q=weak_q
    )


def _get_indexes(arr, virtual_indexes, valid_values_count):
    """The indexes of the order statistics on each side of every virtual index."""
    previous_indexes = np.floor(virtual_indexes, out=...)
    next_indexes = np.add(previous_indexes, 1, out=...)
    indexes_above_bounds = virtual_indexes >= valid_values_count - 1
    # When indexes is above max index, take the max value of the array
    if indexes_above_bounds.any():
        previous_indexes[indexes_above_bounds] = -1
        next_indexes[indexes_above_bounds] = -1
    # When indexes is below min index, take the min value of the array
    indexes_below_bounds = virtual_indexes < 0
    if indexes_below_bounds.any():
        previous_indexes[indexes_below_bounds] = 0
        next_indexes[indexes_below_bounds] = 0
    if np.issubdtype(arr.dtype, np.inexact):
        # After the sort, slices having NaNs will have for last element a NaN
        virtual_indexes_nans = np.isnan(virtual_indexes)
        if virtual_indexes_nans.any():
            previous_indexes[virtual_indexes_nans] = -1
            next_indexes[virtual_indexes_nans] = -1
    previous_indexes = previous_indexes.astype(np.intp)
    next_indexes = next_indexes.astype(np.intp)
    return previous_indexes, next_indexes


def _quantile(arr, quantiles, axis=-1, method="linear", out=None, weights=None, weak_q=False):
    # --- Setup
    arr = np.asanyarray(arr)
    values_count = arr.shape[axis]
    # The dimensions of `q` are prepended to the output shape, so we need the
    # axis being sampled from `arr` to be last.
    if axis != 0:  # But moveaxis is slow, so only call it if necessary.
        arr = np.moveaxis(arr, axis, destination=0)
    supports_nans = np.issubdtype(arr.dtype, np.inexact) or arr.dtype.kind in "Mm"

    if weights is None:
        # --- Computation of indexes
        # Index where to find the value in the sorted array.
        # Virtual because it is a floating point value, not a valid index.
        # The nearest neighbours are used for interpolation
        try:
            method_props = _QuantileMethods[method]
        except KeyError:
            raise ValueError(
                f"{method!r} is not a valid method. Use one of: {_QuantileMethods.keys()}"
            ) from None
        virtual_indexes = method_props["get_virtual_index"](values_count, quantiles)
        virtual_indexes = np.asanyarray(virtual_indexes)

        if method_props["fix_gamma"] is None:
            supports_integers = True
        else:
            int_virtual_indices = np.issubdtype(virtual_indexes.dtype, np.integer)
            supports_integers = method == "linear" and int_virtual_indices

        if supports_integers:
            # No interpolation needed, take the points along axis
            if supports_nans:
                # may contain nan, which would sort to the end
                arr.partition(np.concatenate((virtual_indexes.ravel(), [-1])), axis=0)
                slices_having_nans = np.isnan(arr[-1, ...])
            else:
                # cannot contain nan
                arr.partition(virtual_indexes.ravel(), axis=0)
                slices_having_nans = np.array(False, dtype=bool)
            result = np.take(arr, virtual_indexes, axis=0, out=out)
        else:
            previous_indexes, next_indexes = _get_indexes(arr, virtual_indexes, values_count)
            # --- Sorting
            arr.partition(
                np.unique(
                    np.concatenate(([0, -1], previous_indexes.ravel(), next_indexes.ravel()))
                ),
                axis=0,
            )
            if supports_nans:
                slices_having_nans = np.isnan(arr[-1, ...])
            else:
                slices_having_nans = None
            # --- Get values from indexes
            previous = arr[previous_indexes]
            next = arr[next_indexes]
            # --- Linear interpolation
            gamma = _get_gamma(virtual_indexes, previous_indexes, method_props)
            if weak_q:
                gamma = float(gamma)
            else:
                result_shape = virtual_indexes.shape + (1,) * (arr.ndim - 1)
                gamma = gamma.reshape(result_shape)
            result = _lerp(previous, next, gamma, out=out)
    else:
        # Weighted case
        # This implements method="inverted_cdf", the only supported weighted
        # method, which needs to sort anyway.
        weights = np.asanyarray(weights)
        if axis != 0:
            weights = np.moveaxis(weights, axis, destination=0)
        index_array = np.argsort(arr, axis=0)

        # arr = arr[index_array, ...]  # but this adds trailing dimensions of
        # 1.
        arr = np.take_along_axis(arr, index_array, axis=0)
        if weights.shape == arr.shape:
            weights = np.take_along_axis(weights, index_array, axis=0)
        else:
            # weights is 1d
            weights = weights.reshape(-1)[index_array, ...]

        if supports_nans:
            # may contain nan, which would sort to the end
            slices_having_nans = np.isnan(arr[-1, ...])
        else:
            # cannot contain nan
            slices_having_nans = np.array(False, dtype=bool)

        # We use the weights to calculate the empirical cumulative
        # distribution function cdf
        cdf = weights.cumsum(axis=0, dtype=np.float64)
        cdf /= cdf[-1, ...]  # normalization to 1
        if np.isnan(cdf[-1]).any():
            # Above calculations should normally warn for the zero/inf case.
            raise ValueError("Weights included NaN, inf or were all zero.")
        # Search index i such that
        #   sum(weights[j], j=0..i-1) < quantile <= sum(weights[j], j=0..i)
        # is then equivalent to
        #   cdf[i-1] < quantile <= cdf[i]
        # Unfortunately, searchsorted only accepts 1-d arrays as first
        # argument, so we will need to iterate over dimensions.

        # Without the following cast, searchsorted can return surprising
        # results, e.g.
        #   np.searchsorted(np.array([0.2, 0.4, 0.6, 0.8, 1.]),
        #                   np.array(0.4, dtype=np.float32), side="left")
        # returns 2 instead of 1 because 0.4 is not binary representable.
        if quantiles.dtype.kind == "f":
            cdf = cdf.astype(quantiles.dtype)
        # Weights must be non-negative, so we might have zero weights at the
        # beginning leading to some leading zeros in cdf. The call to
        # np.searchsorted for quantiles=0 will then pick the first element,
        # but should pick the first one larger than zero. We
        # therefore simply set 0 values in cdf to -1.
        if np.any(cdf[0, ...] == 0):
            cdf[cdf == 0] = -1

        def find_cdf_1d(arr, cdf):
            indices = np.searchsorted(cdf, quantiles, side="left")
            # We might have reached the maximum with i = len(arr), e.g. for
            # quantiles = 1, and need to cut it to len(arr) - 1.
            indices = np.minimum(indices, values_count - 1)
            result = np.take(arr, indices, axis=0)
            return result

        r_shape = arr.shape[1:]
        if quantiles.ndim > 0:
            r_shape = quantiles.shape + r_shape
        if out is None:
            result = np.empty_like(arr, shape=r_shape)
        else:
            if out.shape != r_shape:
                msg = (
                    f"Wrong shape of argument 'out', shape={r_shape} is required; "
                    f"got shape={out.shape}."
                )
                raise ValueError(msg)
            result = out

        # See apply_along_axis, which we do for axis=0. Note that Ni = (,)
        # always, so we remove it here.
        Nk = arr.shape[1:]
        for kk in np.ndindex(Nk):
            result[(...,) + kk] = find_cdf_1d(arr[np.s_[:,] + kk], cdf[np.s_[:,] + kk])

        # Make result the same as in unweighted inverted_cdf.
        if result.shape == () and result.dtype == np.dtype("O"):
            result = result.item()

    if np.any(slices_having_nans):
        if result.ndim == 0 and out is None:
            # can't write to a scalar, but indexing will be correct
            result = arr[-1]
        else:
            np.copyto(result, arr[-1, ...], where=slices_having_nans)
    return result


def cov(
    m, y=None, rowvar=True, bias=False, ddof=None, fweights=None, aweights=None, *, dtype=None
):
    if ddof is not None and ddof != int(ddof):
        raise ValueError("ddof must be integer")
    m = np.asarray(m)
    if m.ndim > 2:
        raise ValueError("m has more than 2 dimensions")
    if y is not None:
        y = np.asarray(y)
        if y.ndim > 2:
            raise ValueError("y has more than 2 dimensions")
    if dtype is None:
        if y is None:
            dtype = np.result_type(m, np.float64)
        else:
            dtype = np.result_type(m, y, np.float64)
    X = np.array(m, ndmin=2, dtype=dtype)
    if not rowvar and m.ndim != 1:
        X = X.T
    if X.shape[0] == 0:
        return np.array([]).reshape(0, 0)
    if y is not None:
        y = np.array(y, copy=None, ndmin=2, dtype=dtype)
        if not rowvar and y.shape[0] != 1:
            y = y.T
        X = np.concatenate((X, y), axis=0)
    if ddof is None:
        if bias == 0:
            ddof = 1
        else:
            ddof = 0
    w = None
    if fweights is not None:
        fweights = np.asarray(fweights, dtype=float)
        if not np.all(fweights == np.around(fweights)):
            raise TypeError("fweights must be integer")
        if fweights.ndim > 1:
            raise RuntimeError("cannot handle multidimensional fweights")
        if fweights.shape[0] != X.shape[1]:
            raise RuntimeError("incompatible numbers of samples and fweights")
        if any(fweights < 0):
            raise ValueError("fweights cannot be negative")
        w = fweights
    if aweights is not None:
        aweights = np.asarray(aweights, dtype=float)
        if aweights.ndim > 1:
            raise RuntimeError("cannot handle multidimensional aweights")
        if aweights.shape[0] != X.shape[1]:
            raise RuntimeError("incompatible numbers of samples and aweights")
        if any(aweights < 0):
            raise ValueError("aweights cannot be negative")
        if w is None:
            w = aweights
        else:
            w *= aweights
    avg, w_sum = np.average(X, axis=1, weights=w, returned=True)
    w_sum = w_sum[0]
    if w is None:
        fact = X.shape[1] - ddof
    elif ddof == 0:
        fact = w_sum
    elif aweights is None:
        fact = w_sum - ddof
    else:
        fact = w_sum - ddof * sum(w * aweights) / w_sum
    if fact <= 0:
        warnings.warn("Degrees of freedom <= 0 for slice", RuntimeWarning, stacklevel=2)
        fact = 0.0
    X -= avg[:, None]
    if w is None:
        X_T = X.T
    else:
        X_T = (X * w).T
    c = np.dot(X, X_T.conj())
    c *= np.true_divide(1, fact)
    return c.squeeze()


def corrcoef(x, y=None, rowvar=True, *, dtype=None):
    c = cov(x, y, rowvar, dtype=dtype)
    try:
        d = np.diag(c)
    except ValueError:
        return c / c
    stddev = np.sqrt(d.real)
    c /= stddev[:, None]
    c /= stddev[None, :]
    np.clip(c.real, -1, 1, out=c.real)
    if np.iscomplexobj(c):
        np.clip(c.imag, -1, 1, out=c.imag)
    return c
