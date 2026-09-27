"""shellsim's ``scipy.spatial``: the ``distance`` module and the deprecated Minkowski helpers.

Spatial data structures and computational geometry (``KDTree``, ``ConvexHull``, ``Delaunay``,
``Voronoi``, ...) are not modeled and raise ``NotImplementedError`` when accessed.
"""

import warnings

import numpy as np

from scipy.spatial import distance

__all__ = ["distance", "distance_matrix", "minkowski_distance", "minkowski_distance_p"]

_UNSUPPORTED = {
    "ConvexHull",
    "Delaunay",
    "HalfspaceIntersection",
    "KDTree",
    "QhullError",
    "Rectangle",
    "SphericalVoronoi",
    "Voronoi",
    "cKDTree",
    "ckdtree",
    "convex_hull_plot_2d",
    "delaunay_plot_2d",
    "geometric_slerp",
    "kdtree",
    "procrustes",
    "qhull",
    "transform",
    "tsearch",
    "voronoi_plot_2d",
}


def __getattr__(name):
    if name in _UNSUPPORTED:
        raise NotImplementedError(f"scipy.spatial.{name} is not supported by shellsim's SciPy")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def _deprecated(name, replacement):
    warnings.warn(
        f"`{name}` is deprecated in favor of `scipy.spatial.distance.{replacement}` as of SciPy "
        "1.18.0 and will be removed in SciPy 1.20.0.",
        DeprecationWarning,
        stacklevel=3,
    )


def _minkowski_p(x, y, p):
    diff = np.abs(np.asarray(x) - np.asarray(y))
    if p == np.inf:
        return np.max(diff, axis=-1)
    if p == 1:
        return np.sum(diff, axis=-1)
    return np.sum(diff**p, axis=-1)


def _minkowski(x, y, p):
    if p == np.inf or p == 1:
        return _minkowski_p(x, y, p)
    return _minkowski_p(x, y, p) ** (1.0 / p)


def minkowski_distance_p(x, y, p=2.0):
    """``sum(|x - y|**p)`` along the last axis of the broadcast operands; ``max`` for inf."""
    _deprecated("minkowski_distance_p", "minkowski")
    return _minkowski_p(x, y, p)


def minkowski_distance(x, y, p=2.0):
    """The Minkowski distance along the last axis of the broadcast operands."""
    _deprecated("minkowski_distance", "minkowski")
    return _minkowski(x, y, p)


def distance_matrix(x, y, p=2.0, threshold=1000000):
    """The ``(m, n)`` matrix of Minkowski distances between the rows of ``x`` and of ``y``."""
    _deprecated("distance_matrix", "cdist")
    x = np.asarray(x)
    y = np.asarray(y)
    if x.shape[1] != y.shape[1]:
        raise ValueError(
            f"x contains {x.shape[1]}-dimensional vectors but y contains "
            f"{y.shape[1]}-dimensional vectors"
        )
    return _minkowski(x[:, np.newaxis, :], y[np.newaxis, :, :], p)
