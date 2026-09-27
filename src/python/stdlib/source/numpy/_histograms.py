"""``histogram``, ``histogram_bin_edges`` and ``digitize``.

``histogram`` and its bin-width estimators follow ``numpy/lib/_histograms_impl.py``: uniform bins
compute each element's bin arithmetically and then correct it against the float edges, and
other bins count through sorted search. ``digitize`` follows
``numpy/lib/_function_base_impl.py``, with ``_monotonicity`` ported from
``numpy/_core/src/multiarray/compiled_base.c``. ``histogram2d`` and ``histogramdd`` are not
provided.
"""

import operator
import warnings

import numpy as np

_range = range


def _ptp(x):
    """Peak-to-peak value of x, computed without integer overflow."""
    return _unsigned_subtract(x.max(), x.min())


def _hist_bin_sqrt(x, range):
    del range  # unused
    return _ptp(x) / np.sqrt(x.size)


def _hist_bin_sturges(x, range):
    del range  # unused
    return _ptp(x) / (np.log2(x.size) + 1.0)


def _hist_bin_rice(x, range):
    del range  # unused
    return _ptp(x) / (2.0 * x.size ** (1.0 / 3))


def _hist_bin_scott(x, range):
    del range  # unused
    return (24.0 * np.pi**0.5 / x.size) ** (1.0 / 3.0) * np.std(x)


def _hist_bin_stone(x, range):
    n = x.size
    ptp_x = _ptp(x)
    if n <= 1 or ptp_x == 0:
        return 0

    def jhat(nbins):
        hh = ptp_x / nbins
        p_k = np.histogram(x, bins=nbins, range=range)[0] / n
        return (2 - (n + 1) * p_k.dot(p_k)) / hh

    nbins_upper_bound = max(100, int(np.sqrt(n)))
    nbins = min(_range(1, nbins_upper_bound + 1), key=jhat)
    if nbins == nbins_upper_bound:
        warnings.warn(
            "The number of bins estimated may be suboptimal.", RuntimeWarning, stacklevel=3
        )
    return ptp_x / nbins


def _hist_bin_doane(x, range):
    del range  # unused
    if x.size > 2:
        sg1 = np.sqrt(6.0 * (x.size - 2) / ((x.size + 1.0) * (x.size + 3)))
        sigma = np.std(x)
        if sigma > 0.0:
            # These three operations add up to
            # g1 = np.mean(((x - np.mean(x)) / sigma)**3)
            # but use only one temp array instead of three
            temp = x - np.mean(x)
            np.true_divide(temp, sigma, temp)
            np.power(temp, 3, temp)
            g1 = np.mean(temp)
            return _ptp(x) / (1.0 + np.log2(x.size) + np.log2(1.0 + np.absolute(g1) / sg1))
    return 0.0


def _hist_bin_fd(x, range):
    del range  # unused
    iqr = np.subtract(*np.percentile(x, [75, 25]))
    return 2.0 * iqr * x.size ** (-1.0 / 3.0)


def _hist_bin_auto(x, range):
    fd_bw = _hist_bin_fd(x, range)
    sturges_bw = _hist_bin_sturges(x, range)
    sqrt_bw = _hist_bin_sqrt(x, range)
    # heuristic to limit the maximal number of bins
    fd_bw_corrected = max(fd_bw, sqrt_bw / 2)
    return min(fd_bw_corrected, sturges_bw)


_hist_bin_selectors = {
    "stone": _hist_bin_stone,
    "auto": _hist_bin_auto,
    "doane": _hist_bin_doane,
    "fd": _hist_bin_fd,
    "rice": _hist_bin_rice,
    "scott": _hist_bin_scott,
    "sqrt": _hist_bin_sqrt,
    "sturges": _hist_bin_sturges,
}


def _ravel_and_check_weights(a, weights):
    a = np.asarray(a)
    # Ensure that the array is a "subtractable" dtype
    if a.dtype == np.bool:
        msg = f"Converting input from {a.dtype} to {np.uint8} for compatibility."
        warnings.warn(msg, RuntimeWarning, stacklevel=3)
        a = a.astype(np.uint8)
    if weights is not None:
        weights = np.asarray(weights)
        if weights.shape != a.shape:
            raise ValueError("weights should have the same shape as a.")
        weights = weights.ravel()
    a = a.ravel()
    return a, weights


def _get_outer_edges(a, range):
    if range is not None:
        first_edge, last_edge = range
        if first_edge > last_edge:
            raise ValueError("max must be larger than min in range parameter.")
        if not (np.isfinite(first_edge) and np.isfinite(last_edge)):
            raise ValueError(f"supplied range of [{first_edge}, {last_edge}] is not finite")
    elif a.size == 0:
        # handle empty arrays. Can't determine range, so use 0-1.
        first_edge, last_edge = 0, 1
    else:
        first_edge, last_edge = a.min(), a.max()
        if not (np.isfinite(first_edge) and np.isfinite(last_edge)):
            raise ValueError(f"autodetected range of [{first_edge}, {last_edge}] is not finite")
    # expand empty range to avoid divide by zero
    if first_edge == last_edge:
        first_edge = first_edge - 0.5
        last_edge = last_edge + 0.5
    return first_edge, last_edge


def _unsigned_subtract(a, b):
    """Subtract, giving an unsigned result for signed integers so the difference cannot wrap."""
    signed_to_unsigned = {
        np.byte: np.ubyte,
        np.short: np.ushort,
        np.intc: np.uintc,
        np.int_: np.uint,
        np.longlong: np.ulonglong,
    }
    dt = np.result_type(a, b)
    try:
        unsigned_dt = signed_to_unsigned[dt.type]
    except KeyError:
        return np.subtract(a, b, dtype=dt)
    # we know the inputs are integers, and we are deliberately casting
    # signed to unsigned.  The input may be negative python integers so
    # ensure we pass in arrays with the initial dtype (related to NEP 50).
    return np.subtract(
        np.asarray(a, dtype=dt), np.asarray(b, dtype=dt), casting="unsafe", dtype=unsigned_dt
    )


def _get_bin_edges(a, bins, range, weights):
    # parse the overloaded bins argument
    n_equal_bins = None
    bin_edges = None
    if isinstance(bins, str):
        bin_name = bins
        # if `bins` is a string for an automatic method,
        # this will replace it with the number of bins calculated
        if bin_name not in _hist_bin_selectors:
            raise ValueError(f"{bin_name!r} is not a valid estimator for `bins`")
        if weights is not None:
            raise TypeError(
                "Automated estimation of the number of bins is not supported for weighted data"
            )
        first_edge, last_edge = _get_outer_edges(a, range)
        # truncate the range if needed
        if range is not None:
            keep = a >= first_edge
            keep &= a <= last_edge
            if not np.logical_and.reduce(keep):
                a = a[keep]
        if a.size == 0:
            n_equal_bins = 1
        else:
            # Do not call selectors on empty arrays
            width = _hist_bin_selectors[bin_name](a, (first_edge, last_edge))
            if width:
                if np.issubdtype(a.dtype, np.integer) and width < 1:
                    width = 1
                delta = _unsigned_subtract(last_edge, first_edge)
                n_equal_bins = int(np.ceil(delta / width))
            else:
                # Width can be zero for some estimators, e.g. FD when
                # the IQR of the data is zero.
                n_equal_bins = 1
    elif np.ndim(bins) == 0:
        try:
            n_equal_bins = operator.index(bins)
        except TypeError as e:
            raise TypeError("`bins` must be an integer, a string, or an array") from e
        if n_equal_bins < 1:
            raise ValueError("`bins` must be positive, when an integer")
        first_edge, last_edge = _get_outer_edges(a, range)
    elif np.ndim(bins) == 1:
        bin_edges = np.asarray(bins)
        if np.any(bin_edges[:-1] > bin_edges[1:]):
            raise ValueError("`bins` must increase monotonically, when an array")
    else:
        raise ValueError("`bins` must be 1d, when an array")

    if n_equal_bins is not None:
        # gh-10322 means that type resolution rules are dependent on array
        # shapes. To avoid this causing problems, we pick a type now and stick
        # with it throughout.
        bin_type = np.result_type(first_edge, last_edge, a)
        if np.issubdtype(bin_type, np.integer):
            bin_type = np.result_type(bin_type, float)
        # bin edges must be computed
        bin_edges = np.linspace(
            first_edge, last_edge, n_equal_bins + 1, endpoint=True, dtype=bin_type
        )
        if np.any(bin_edges[:-1] >= bin_edges[1:]):
            raise ValueError(
                f"Too many bins for data range. Cannot create {n_equal_bins} finite-sized bins."
            )
        return bin_edges, (first_edge, last_edge, n_equal_bins)
    return bin_edges, None


def _search_sorted_inclusive(a, v):
    """Like ``searchsorted``, but where the last item in ``v`` is placed on the right."""
    return np.concatenate((a.searchsorted(v[:-1], "left"), a.searchsorted(v[-1:], "right")))


def histogram_bin_edges(a, bins=10, range=None, weights=None):
    a, weights = _ravel_and_check_weights(a, weights)
    bin_edges, _ = _get_bin_edges(a, bins, range, weights)
    return bin_edges


def histogram(a, bins=10, range=None, density=None, weights=None):
    a, weights = _ravel_and_check_weights(a, weights)
    bin_edges, uniform_bins = _get_bin_edges(a, bins, range, weights)
    # Histogram is an integer or a float array depending on the weights.
    if weights is None:
        ntype = np.dtype(np.intp)
    else:
        ntype = weights.dtype
    # We set a block size, as this allows us to iterate over chunks when
    # computing histograms, to minimize memory usage.
    BLOCK = 65536
    # The fast path uses bincount, but that only works for certain types
    # of weight
    simple_weights = (
        weights is None
        or np.can_cast(weights.dtype, np.double)
        or np.can_cast(weights.dtype, complex)
    )
    if uniform_bins is not None and simple_weights:
        # Fast algorithm for equal bins
        # We now convert values of a to bin indices, under the assumption of
        # equal bin widths (which is valid here).
        first_edge, last_edge, n_equal_bins = uniform_bins
        # Initialize empty histogram
        n = np.zeros(n_equal_bins, ntype)
        # Pre-compute histogram scaling factor
        norm_numerator = n_equal_bins
        norm_denom = _unsigned_subtract(last_edge, first_edge)
        # We iterate over blocks here for two reasons: the first is that for
        # large arrays, it is actually faster (for example for a 10^8 array it
        # is 2x as fast) and it results in a memory footprint 3x lower in the
        # limit of large arrays.
        for i in _range(0, len(a), BLOCK):
            tmp_a = a[i : i + BLOCK]
            if weights is None:
                tmp_w = None
            else:
                tmp_w = weights[i : i + BLOCK]
            # Only include values in the right range
            keep = tmp_a >= first_edge
            keep &= tmp_a <= last_edge
            if not np.logical_and.reduce(keep):
                tmp_a = tmp_a[keep]
                if tmp_w is not None:
                    tmp_w = tmp_w[keep]
            # This cast ensures no type promotions occur below, which gh-10322
            # make unpredictable. Getting it wrong leads to precision errors
            # like gh-8123.
            tmp_a = tmp_a.astype(bin_edges.dtype, copy=False)
            # Compute the bin indices, and for values that lie exactly on
            # last_edge we need to subtract one
            f_indices = _unsigned_subtract(tmp_a, first_edge) / norm_denom * norm_numerator
            indices = f_indices.astype(np.intp)
            indices[indices == n_equal_bins] -= 1
            # The index computation is not guaranteed to give exactly
            # consistent results within ~1 ULP of the bin edges.
            decrement = tmp_a < bin_edges[indices]
            indices[decrement] -= 1
            # The last bin includes the right edge. The other bins do not.
            increment = (tmp_a >= bin_edges[indices + 1]) & (indices != n_equal_bins - 1)
            indices[increment] += 1
            # We now compute the histogram using bincount
            if ntype.kind == "c":
                n.real += np.bincount(indices, weights=tmp_w.real, minlength=n_equal_bins)
                n.imag += np.bincount(indices, weights=tmp_w.imag, minlength=n_equal_bins)
            else:
                n += np.bincount(indices, weights=tmp_w, minlength=n_equal_bins).astype(ntype)
    else:
        # Compute via cumulative histogram
        cum_n = np.zeros(bin_edges.shape, ntype)
        if weights is None:
            for i in _range(0, len(a), BLOCK):
                sa = np.sort(a[i : i + BLOCK])
                cum_n += _search_sorted_inclusive(sa, bin_edges)
        else:
            zero = np.zeros(1, dtype=ntype)
            for i in _range(0, len(a), BLOCK):
                tmp_a = a[i : i + BLOCK]
                tmp_w = weights[i : i + BLOCK]
                sorting_index = np.argsort(tmp_a)
                sa = tmp_a[sorting_index]
                sw = tmp_w[sorting_index]
                cw = np.concatenate((zero, sw.cumsum()))
                bin_index = _search_sorted_inclusive(sa, bin_edges)
                cum_n += cw[bin_index]
        n = np.diff(cum_n)
    if density:
        # this will fail if weights are not a numeric type
        db = np.array(np.diff(bin_edges), float)
        return n / db / n.sum(), bin_edges
    return n, bin_edges


def _monotonicity(bins):
    """1 if ``bins`` never decreases, -1 if it never increases, and 0 otherwise."""
    bins = np.asarray(bins, dtype=np.float64)
    if bins.ndim == 0:
        raise ValueError("object of too small depth for desired array")
    if bins.ndim > 1:
        raise ValueError("object too deep for desired array")
    values = bins.tolist()
    if not values:
        return 1
    last = values[0]
    # Skip repeated values at the beginning of the array
    i = 1
    while i < len(values) and values[i] == last:
        i += 1
    if i == len(values):
        return 1
    following = values[i]
    if last < following:
        for value in values[i + 1 :]:
            if following > value:
                return 0
            following = value
        return 1
    for value in values[i + 1 :]:
        if following < value:
            return 0
        following = value
    return -1


def digitize(x, bins, right=False):
    x = np.asarray(x)
    bins = np.asarray(bins)
    # here for compatibility, searchsorted below is happy to take this
    if np.issubdtype(x.dtype, np.complexfloating):
        raise TypeError("x may not be complex")
    mono = _monotonicity(bins)
    if mono == 0:
        raise ValueError("bins must be monotonically increasing or decreasing")
    # this is backwards because the arguments below are swapped
    side = "left" if right else "right"
    if mono == -1:
        # reverse the bins, and invert the results
        return len(bins) - np.searchsorted(bins[::-1], x, side=side)
    return np.searchsorted(bins, x, side=side)
