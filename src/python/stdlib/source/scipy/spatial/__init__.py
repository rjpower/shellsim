"""shellsim's ``scipy.spatial``: the ``distance`` module.

Spatial data structures and computational geometry (``KDTree``, ``ConvexHull``, ``Delaunay``,
``Voronoi``, ...) and the deprecated Minkowski helpers (``distance_matrix``,
``minkowski_distance``, ``minkowski_distance_p``) are not provided.
"""

from scipy.spatial import distance

__all__ = ["distance"]
