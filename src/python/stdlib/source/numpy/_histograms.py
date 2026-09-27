"""``histogram``, its bin-edge computation, and ``digitize``: counting values into ordered bins.

``histogram`` places each value with :func:`numpy.searchsorted` against the bin edges and tallies
the result with the native :func:`numpy.bincount`, so the only Python-level looping is over the
handful of edges, never over the data.
"""

import numpy as np

__all__ = ["digitize", "histogram", "histogram_bin_edges"]


def histogram_bin_edges(a, bins=10, range=None, weights=None):
    """The edges :func:`histogram` would use for `a`, `bins` and `range`."""
    a = np.asanyarray(a)
    if isinstance(bins, str):
        raise NotImplementedError(f"np.histogram_bin_edges(bins={bins!r}) is not supported")
    bins_array = np.asanyarray(bins)
    if bins_array.ndim == 1:
        edges = bins_array.astype(np.float64)
        if edges.size > 1 and bool(np.any(edges[1:] < edges[:-1])):
            raise ValueError("bins must increase monotonically, when an array")
        return edges
    count = int(bins)
    if count < 1:
        raise ValueError("`bins` must be positive, when an integer")
    if range is not None:
        low, high = float(range[0]), float(range[1])
    elif a.size == 0:
        low, high = 0.0, 1.0
    else:
        low, high = float(np.min(a)), float(np.max(a))
    if low == high:
        low, high = low - 0.5, high + 0.5
    return np.linspace(low, high, count + 1)


def histogram(a, bins=10, range=None, density=False, weights=None):
    """Counts (or, if `density`, a probability density) of `a` over `bins`, and the bin edges."""
    a = np.asanyarray(a).reshape(-1)
    if weights is not None:
        weights = np.asanyarray(weights).reshape(-1)
    edges = histogram_bin_edges(a, bins, range, weights)
    bin_count = edges.size - 1
    positions = np.searchsorted(edges, a, side="right") - 1
    # A value equal to the last edge belongs in the last bin, not past it.
    positions = np.where(a == edges[-1], bin_count - 1, positions)
    in_range = (positions >= 0) & (positions < bin_count)
    kept = positions[in_range]
    if weights is None:
        counts = np.bincount(kept, minlength=bin_count)[:bin_count]
    else:
        counts = np.bincount(kept, weights=weights[in_range], minlength=bin_count)[:bin_count]
    if density:
        widths = np.diff(edges)
        counts = counts / (counts.sum() * widths)
    return counts, edges


def digitize(x, bins, right=False):
    """The index of the bin (from `bins`, increasing or decreasing) each value of `x` falls in."""
    x = np.asanyarray(x)
    bins = np.asanyarray(bins)
    side = "left" if right else "right"
    if bins.size < 2 or bool(bins[-1] >= bins[0]):
        return np.searchsorted(bins, x, side=side)
    return bins.size - np.searchsorted(bins[::-1], x, side=side)
