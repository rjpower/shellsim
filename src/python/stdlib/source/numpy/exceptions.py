"""numpy.exceptions: the warning and error types NumPy raises.

``AxisError`` and ``ComplexWarning`` are backed by native exception types registered under a
leading-underscore name on the ``numpy`` module (``_AxisError`` and ``_ComplexWarning``); the
interpreter's ufunc and casting code raises them directly without going through this module. This
module exposes them under their public names and defines the remaining, pure-Python warning
types. Casting and dtype-promotion failures raise a plain ``TypeError`` rather than a dedicated
subclass: shellsim matches the exception type ordinary code catches, not NumPy's exact class
hierarchy for internal failures.
"""

from numpy import _AxisError as AxisError
from numpy import _ComplexWarning as ComplexWarning

__all__ = [
    "AxisError",
    "ComplexWarning",
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
