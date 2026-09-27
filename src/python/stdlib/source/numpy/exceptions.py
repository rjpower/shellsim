"""numpy.exceptions: the warning and error types NumPy raises.

``AxisError``, ``ComplexWarning``, and ``DTypePromotionError`` are backed by native exception
types registered under a leading-underscore name on the ``numpy`` module (``_AxisError`` and
friends); the interpreter's ufunc and casting code raises them directly without going through this
module. This module exposes them under their public names and defines the remaining, pure-Python
warning types.
"""

from numpy import _AxisError as AxisError
from numpy import _ComplexWarning as ComplexWarning
from numpy import _DTypePromotionError as DTypePromotionError

__all__ = [
    "AxisError",
    "ComplexWarning",
    "DTypePromotionError",
    "ModuleDeprecationWarning",
    "RankWarning",
    "TooHardError",
    "VisibleDeprecationWarning",
]


class ModuleDeprecationWarning(DeprecationWarning):
    """A feature endpoint of an entire module is deprecated."""


class RankWarning(RuntimeWarning):
    """A matrix or polynomial fit is badly conditioned or rank-deficient."""


class TooHardError(RuntimeError):
    """shellsim's NumPy gives up on a computation it cannot complete simply."""


class VisibleDeprecationWarning(UserWarning):
    """A deprecation warning that is visible by default, unlike ``DeprecationWarning``."""
