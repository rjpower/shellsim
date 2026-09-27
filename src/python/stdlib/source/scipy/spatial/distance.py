"""shellsim's ``scipy.spatial.distance``.

Every metric is one row kernel: a vectorized NumPy expression that reduces the last axis of two
broadcast operands. The public vector functions (``euclidean(u, v)``, ...) apply a kernel to
their arguments directly, so like SciPy's they also reduce the last axis of higher-dimensional
input. ``cdist`` applies it to ``XA[:, None]`` against ``XB[None, :]`` and ``pdist`` to the pairs
``np.triu_indices`` selects, both in float64 with floating-point warnings silenced, as SciPy's
compiled loops are. A callable metric is called once per pair instead.

The boolean dissimilarities other than ``hamming`` and ``jaccard`` (``dice``, ``yule``, ...)
and ``directed_hausdorff`` are not modeled and raise ``NotImplementedError``.
"""

import math
import warnings

import numpy as np
from scipy.special import rel_entr

__all__ = [
    "braycurtis",
    "canberra",
    "cdist",
    "chebyshev",
    "cityblock",
    "correlation",
    "cosine",
    "euclidean",
    "is_valid_dm",
    "is_valid_y",
    "jaccard",
    "jensenshannon",
    "mahalanobis",
    "minkowski",
    "num_obs_dm",
    "num_obs_y",
    "pdist",
    "seuclidean",
    "sqeuclidean",
    "squareform",
    "hamming",
]

_UNSUPPORTED = {
    "dice",
    "directed_hausdorff",
    "rogerstanimoto",
    "russellrao",
    "sokalsneath",
    "yule",
}


def __getattr__(name):
    if name in _UNSUPPORTED:
        raise NotImplementedError(f"scipy.spatial.distance.{name} is not supported by shellsim's SciPy")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def _as_float(u):
    """``u`` as an array, with integer and boolean input promoted to float64."""
    u = np.asarray(u)
    return u if np.issubdtype(u.dtype, np.inexact) else u.astype(np.float64)


def _weights(w):
    if w is None:
        return None
    w = np.asarray(w, dtype=np.float64)
    if np.any(w < 0):
        raise ValueError("Input weights should be all non-negative")
    return w


def _pairwise_sum(values):
    """NumPy's summation of the last axis, which SciPy's vector functions use."""
    return np.sum(values, axis=-1)


def _sequential_sum(values):
    """Left-to-right summation of the last axis, as SciPy's compiled pairwise loops sum."""
    if values.shape[-1] == 0:
        return np.sum(values, axis=-1)
    return np.cumsum(values, axis=-1)[..., -1]


# Row kernels. Each reduces the last axis of the broadcast operands `u` and `v`, adding with
# `total`: `_pairwise_sum` or `_sequential_sum`.


def _minkowski(total, u, v, p=2, w=None):
    if p <= 0:
        raise ValueError("p must be greater than 0")
    if np.isinf(p):
        return _chebyshev(total, u, v, w)
    w = _weights(w)
    diff = np.abs(u - v)
    if p == 1:
        return _weighted(total, diff, w)
    if p == 2:
        return np.sqrt(_weighted(total, diff * diff, w))
    return _weighted(total, diff**p, w) ** (1.0 / p)


def _weighted(total, values, w):
    return total(values if w is None else w * values)


def _euclidean(total, u, v, w=None):
    return np.sqrt(_sqeuclidean(total, u, v, w))


def _sqeuclidean(total, u, v, w=None):
    diff = u - v
    return _weighted(total, diff * diff, _weights(w))


def _cityblock(total, u, v, w=None):
    return _weighted(total, np.abs(u - v), _weights(w))


def _chebyshev(total, u, v, w=None):
    diff = np.abs(u - v)
    w = _weights(w)
    if w is not None:
        # Coordinates with zero weight do not take part.
        diff = np.where(w > 0, diff, 0)
    return np.max(diff, axis=-1)


def _seuclidean(total, u, v, V):
    V = np.asarray(V, dtype=np.float64)
    if V.ndim != 1:
        raise ValueError("Input vector should be 1-D.")
    if V.shape[0] != np.shape(u)[-1]:
        raise TypeError("V must be a 1-D array of the same dimension as u and v.")
    diff = u - v
    return np.sqrt(total(diff * diff / V))


def _mahalanobis(total, u, v, VI):
    delta = u - v
    product = delta @ np.asarray(VI, dtype=np.float64)
    return np.sqrt(total(product * delta))


def _correlation(total, u, v, w=None, centered=True):
    w = _weights(w)
    if centered:
        u = u - np.expand_dims(_mean(total, u, w), -1)
        v = v - np.expand_dims(_mean(total, v, w), -1)
    uv = _weighted(total, u * v, w)
    uu = _weighted(total, u * u, w)
    vv = _weighted(total, v * v, w)
    dist = 1.0 - uv / np.sqrt(uu * vv)
    # Rounding can leave the cosine of the angle slightly outside [-1, 1].
    return np.clip(dist, 0.0, 2.0)


def _mean(total, values, w):
    if w is None:
        return total(values) / values.shape[-1]
    return total(w * values) / total(w)


def _cosine(total, u, v, w=None):
    return _correlation(total, u, v, w=w, centered=False)


def _hamming(total, u, v, w=None):
    unequal = u != v
    w = _weights(w)
    if w is None:
        return total(unequal) / unequal.shape[-1]
    return total(w / total(w) * unequal)


def _jaccard(total, u, v, w=None):
    # Jaccard compares which coordinates are nonzero, not their values.
    u = u != 0
    v = v != 0
    nonzero = u | v
    unequal = u ^ v
    w = _weights(w)
    if w is not None:
        nonzero = w * nonzero
        unequal = w * unequal
    count = total(nonzero)
    with np.errstate(invalid="ignore"):
        dist = total(unequal) / count
    # Two all-zero vectors are identical.
    return np.where(count == 0, 0.0, dist)[()]


def _braycurtis(total, u, v, w=None):
    w = _weights(w)
    return _weighted(total, np.abs(u - v), w) / _weighted(total, np.abs(u + v), w)


def _canberra(total, u, v, w=None):
    w = _weights(w)
    numerator = np.abs(u - v)
    denominator = np.abs(u) + np.abs(v)
    with np.errstate(invalid="ignore"):
        # A coordinate where both vectors are zero contributes nothing.
        terms = np.where(denominator == 0, 0.0, numerator / denominator)
    return _weighted(total, terms, w)


def _jensenshannon(total, p, q, base=None):
    p = p / np.expand_dims(total(p), -1)
    q = q / np.expand_dims(total(q), -1)
    m = (p + q) / 2.0
    divergence = total(rel_entr(p, m)) + total(rel_entr(q, m))
    if base is not None:
        divergence = divergence / np.log(base)
    return np.sqrt(divergence / 2.0)


# Public vector functions.


def minkowski(u, v, p=2, w=None):
    """The Minkowski distance ``(sum(w * |u - v|**p))**(1/p)``; ``p=inf`` gives Chebyshev."""
    return _minkowski(_pairwise_sum, _as_float(u), _as_float(v), p, w)


def euclidean(u, v, w=None):
    """The Euclidean distance ``sqrt(sum(w * (u - v)**2))``."""
    return _euclidean(_pairwise_sum, _as_float(u), _as_float(v), w)


def sqeuclidean(u, v, w=None):
    """The squared Euclidean distance ``sum(w * (u - v)**2)``."""
    return _sqeuclidean(_pairwise_sum, _as_float(u), _as_float(v), w)


def cityblock(u, v, w=None):
    """The Manhattan distance ``sum(w * |u - v|)``, in the inputs' dtype."""
    return _cityblock(_pairwise_sum, np.asarray(u), np.asarray(v), w)


def chebyshev(u, v, w=None):
    """The largest coordinate difference ``max(|u - v|)`` over coordinates with nonzero weight."""
    return _chebyshev(_pairwise_sum, np.asarray(u), np.asarray(v), w)


def seuclidean(u, v, V):
    """The Euclidean distance with each coordinate scaled by its variance in ``V``."""
    return _seuclidean(_pairwise_sum, _as_float(u), _as_float(v), V)


def mahalanobis(u, v, VI):
    """``sqrt((u - v) @ VI @ (u - v))`` for an inverse covariance matrix ``VI``."""
    return _mahalanobis(_pairwise_sum, _float64(u), _float64(v), VI)


def correlation(u, v, w=None, centered=True):
    """One minus the (weighted) Pearson correlation of ``u`` and ``v``, clipped to [0, 2]."""
    return _correlation(_pairwise_sum, _float64(u), _float64(v), w, centered)


def cosine(u, v, w=None):
    """One minus the cosine of the angle between ``u`` and ``v``, clipped to [0, 2]."""
    return _cosine(_pairwise_sum, _float64(u), _float64(v), w)


def hamming(u, v, w=None):
    """The (weighted) fraction of coordinates where ``u`` and ``v`` differ."""
    return _hamming(_pairwise_sum, np.asarray(u), np.asarray(v), w)


def jaccard(u, v, w=None):
    """The fraction of coordinates nonzero in either vector where exactly one is nonzero."""
    return _jaccard(_pairwise_sum, np.asarray(u), np.asarray(v), w)


def braycurtis(u, v, w=None):
    """``sum(w * |u - v|) / sum(w * |u + v|)``."""
    return _braycurtis(_pairwise_sum, _float64(u), _float64(v), w)


def canberra(u, v, w=None):
    """``sum(w * |u - v| / (|u| + |v|))``, where coordinates that are zero in both count 0."""
    return _canberra(_pairwise_sum, _float64(u), _float64(v), w)


def jensenshannon(p, q, base=None, *, axis=0, keepdims=False):
    """The Jensen-Shannon distance between probability vectors, normalized along ``axis``."""
    p = _float64(p)
    q = _float64(q)
    for operand in (p, q):
        if not -operand.ndim <= axis < operand.ndim:
            raise np.exceptions.AxisError(axis, operand.ndim)
    result = _jensenshannon(_pairwise_sum, np.moveaxis(p, axis, -1), np.moveaxis(q, axis, -1), base)
    return np.expand_dims(result, axis) if keepdims else result


def _float64(u):
    return np.asarray(u, dtype=np.float64)


# Pairwise distances.

_METRICS = {
    "braycurtis": (_braycurtis, ()),
    "canberra": (_canberra, ()),
    "chebyshev": (_chebyshev, ("chebychev", "cheby", "cheb", "ch")),
    "cityblock": (_cityblock, ("cblock", "cb", "c")),
    "correlation": (_correlation, ("co",)),
    "cosine": (_cosine, ("cos",)),
    "euclidean": (_euclidean, ("euclid", "eu", "e")),
    "hamming": (_hamming, ("matching", "hamm", "ha", "h")),
    "jaccard": (_jaccard, ("jacc", "ja", "j")),
    "jensenshannon": (_jensenshannon, ("js",)),
    "mahalanobis": (_mahalanobis, ("mahal", "mah")),
    "minkowski": (_minkowski, ("mi", "m", "pnorm")),
    "seuclidean": (_seuclidean, ("se", "s")),
    "sqeuclidean": (_sqeuclidean, ("sqe", "sqeuclid")),
}

_ALIASES = {
    alias: name for name, (_, aliases) in _METRICS.items() for alias in (name, *aliases)
}

_UNSUPPORTED_METRICS = {
    "dice",
    "kulczynski1",
    "rogerstanimoto",
    "russellrao",
    "sokalmichener",
    "sokalsneath",
    "yule",
}


def _metric_name(metric):
    name = metric.lower()
    if name.startswith("test_"):
        name = name[len("test_") :]
    if name in _UNSUPPORTED_METRICS:
        raise NotImplementedError(f"the {name!r} distance metric is not supported by shellsim's SciPy")
    if name not in _ALIASES:
        raise ValueError(f"Unknown Distance Metric: {metric}")
    return _ALIASES[name]


def _matrix(X):
    """``X`` as a float64 matrix; the caller has checked that it is 2-D."""
    if not (np.issubdtype(X.dtype, np.number) or X.dtype == np.bool_) or np.iscomplexobj(X):
        raise ValueError(f"Unsupported dtype {X.dtype}")
    return X.astype(np.float64)


def _defaults(name, X, kwargs):
    """Fill the data-dependent parameters ``seuclidean`` and ``mahalanobis`` take from ``X``."""
    if name == "seuclidean":
        if "V" not in kwargs:
            kwargs["V"] = np.var(X, axis=0, ddof=1)
        elif np.asarray(kwargs["V"]).shape != (X.shape[1],):
            raise ValueError(
                "Variance vector V must be of the same dimension as the vectors on which the "
                "distances are computed."
            )
    if name == "mahalanobis" and "VI" not in kwargs:
        kwargs["VI"] = np.linalg.inv(np.cov(X.T))
    return kwargs


def _output(result, out):
    if out is None:
        return result
    if out.shape != result.shape:
        raise ValueError("Output array has incorrect shape.")
    if out.dtype != np.float64:
        raise ValueError("wrong out dtype, expected float64")
    out[...] = result
    return out


def _call_pairs(metric, left, right, kwargs):
    result = np.empty(len(left), dtype=np.float64)
    for index in range(len(left)):
        result[index] = metric(left[index], right[index], **kwargs)
    return result


def cdist(XA, XB, metric="euclidean", *, out=None, **kwargs):
    """Distances between each row of ``XA`` and each row of ``XB``, as an ``(mA, mB)`` array."""
    XA = np.asarray(XA)
    XB = np.asarray(XB)
    for label, matrix in (("XA", XA), ("XB", XB)):
        if matrix.ndim != 2:
            raise ValueError(f"{label} must be a 2-dimensional array.")
    if XA.shape[1] != XB.shape[1]:
        raise ValueError(
            "XA and XB must have the same number of columns (i.e. feature dimension.)"
        )
    shape = (XA.shape[0], XB.shape[0])
    if callable(metric):
        rows = np.repeat(np.arange(shape[0]), shape[1])
        columns = np.tile(np.arange(shape[1]), shape[0])
        result = _call_pairs(metric, _matrix(XA)[rows], _matrix(XB)[columns], kwargs)
        return _output(result.reshape(shape), out)
    if not isinstance(metric, str):
        raise TypeError("2nd argument metric must be a string identifier or a function.")
    XA = _matrix(XA)
    XB = _matrix(XB)
    name = _metric_name(metric)
    kwargs = _defaults(name, np.vstack([XA, XB]), kwargs)
    kernel = _METRICS[name][0]
    with np.errstate(all="ignore"):
        result = kernel(_sequential_sum, XA[:, None, :], XB[None, :, :], **kwargs)
    return _output(np.asarray(result, dtype=np.float64).reshape(shape), out)


def pdist(X, metric="euclidean", *, out=None, **kwargs):
    """The condensed distance vector: ``d(X[i], X[j])`` for ``i < j`` in row-major order."""
    X = np.asarray(X)
    if X.ndim != 2:
        raise ValueError(f"A 2-dimensional array must be passed. (Shape was {X.shape}).")
    rows, columns = np.triu_indices(X.shape[0], 1)
    if callable(metric):
        X = _matrix(X)
        return _output(_call_pairs(metric, X[rows], X[columns], kwargs), out)
    if not isinstance(metric, str):
        raise TypeError("2nd argument metric must be a string identifier or a function.")
    X = _matrix(X)
    name = _metric_name(metric)
    if name == "mahalanobis" and "VI" not in kwargs and X.shape[0] <= X.shape[1]:
        raise ValueError(
            f"The number of observations ({X.shape[0]}) is too small; the covariance matrix is "
            f"singular. For observations with {X.shape[1]} dimensions, at least "
            f"{X.shape[1] + 1} observations are required."
        )
    kwargs = _defaults(name, X, kwargs)
    kernel = _METRICS[name][0]
    with np.errstate(all="ignore"):
        result = kernel(_sequential_sum, X[rows], X[columns], **kwargs)
    return _output(np.asarray(result, dtype=np.float64).reshape(len(rows)), out)


# Condensed and square distance matrices.


def squareform(X, force="no", checks=True):
    """Convert a condensed distance vector to a square matrix, or a square matrix to a vector."""
    X = np.ascontiguousarray(X)
    shape = X.shape
    if force.lower() == "tomatrix" and len(shape) != 1:
        raise ValueError("Forcing 'tomatrix' but input X is not a distance vector.")
    if force.lower() == "tovector" and len(shape) != 2:
        raise ValueError("Forcing 'tovector' but input X is not a distance matrix.")
    if len(shape) == 1:
        if shape[0] == 0:
            return np.zeros((1, 1), dtype=X.dtype)
        n = int(math.ceil(math.sqrt(shape[0] * 2)))
        if n * (n - 1) != shape[0] * 2:
            raise ValueError(
                "Incompatible vector size. It must be a binomial coefficient n choose 2 for some "
                "integer n >= 2."
            )
        matrix = np.zeros((n, n), dtype=X.dtype)
        rows, columns = np.triu_indices(n, 1)
        matrix[rows, columns] = X
        matrix[columns, rows] = X
        return matrix
    if len(shape) == 2:
        if shape[0] != shape[1]:
            raise ValueError("The matrix argument must be square.")
        if checks:
            is_valid_dm(X, throw=True, name="X")
        if shape[0] <= 1:
            return np.array([], dtype=X.dtype)
        return X[np.triu_indices(shape[0], 1)]
    raise ValueError(
        "The first argument must be one or two dimensional array. A "
        f"{len(shape)}-dimensional array is not permitted"
    )


def _invalid(message, throw, warning):
    if throw:
        raise ValueError(message)
    if warning:
        warnings.warn(message, stacklevel=3)
    return False


def is_valid_dm(D, tol=0.0, throw=False, name="D", warning=False):
    """Whether ``D`` is a square, symmetric matrix with a zero diagonal, within ``tol``."""
    D = np.asarray(D)
    label = f"Distance matrix {name!r}" if name else "Distance matrix"
    if D.ndim != 2:
        return _invalid(f"{label} must have shape=2 (i.e. be two-dimensional).", throw, warning)
    if D.shape[0] != D.shape[1]:
        return _invalid(f"{label} must be square.", throw, warning)
    if tol == 0.0:
        if not np.array_equal(D, D.T):
            return _invalid(f"{label} must be symmetric.", throw, warning)
        if np.any(np.diagonal(D) != 0):
            return _invalid(f"{label} diagonal must be zero.", throw, warning)
        return True
    if np.any(np.abs(D - D.T) > tol):
        return _invalid(f"{label} must be symmetric within tolerance {tol:5.5f}.", throw, warning)
    if np.any(np.abs(np.diagonal(D)) > tol):
        message = f"{label} diagonal must be close to zero within tolerance {tol:5.5f}."
        return _invalid(message, throw, warning)
    return True


def is_valid_y(y, warning=False, throw=False, name=None):
    """Whether ``y`` is a condensed distance vector: 1-D with ``n * (n - 1) / 2`` entries."""
    y = np.asarray(y)
    label = f"condensed distance matrix {name!r}" if name else "condensed distance matrix"
    if y.ndim != 1:
        message = f"{label[0].upper()}{label[1:]} must have shape=1 (i.e. be one-dimensional)."
        return _invalid(message, throw, warning)
    n = int(math.ceil(math.sqrt(y.shape[0] * 2)))
    if n * (n - 1) != y.shape[0] * 2:
        message = (
            f"Length n of {label} must be a binomial coefficient, i.e. there must be a k such "
            "that (k \\choose 2)=n)!"
        )
        return _invalid(message, throw, warning)
    return True


def num_obs_dm(d):
    """The number of observations a square distance matrix describes."""
    d = np.asarray(d)
    is_valid_dm(d, tol=np.inf, throw=True, name="d")
    return d.shape[0]


def num_obs_y(Y):
    """The number of observations a condensed distance vector describes."""
    Y = np.asarray(Y)
    is_valid_y(Y, throw=True, name="Y")
    if Y.shape[0] == 0:
        raise ValueError(
            "The number of observations cannot be determined on an empty distance matrix."
        )
    return int(math.ceil(math.sqrt(Y.shape[0] * 2)))
