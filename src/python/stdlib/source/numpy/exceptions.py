"""NumPy's exception and warning classes (``numpy.exceptions``)."""

from _numpy import _AxisError as AxisError
from _numpy import _ComplexWarning as ComplexWarning
from _numpy import _DTypePromotionError as DTypePromotionError
from _numpy import _UFuncTypeError as UFuncTypeError


class VisibleDeprecationWarning(UserWarning):
    pass


class RankWarning(RuntimeWarning):
    pass


class TooHardError(RuntimeError):
    pass
