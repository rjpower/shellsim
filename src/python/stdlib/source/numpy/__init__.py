"""shellsim's NumPy.

Arrays, dtypes, ufuncs and most functions are native and live in the ``_numpy*`` modules; this
package re-exports them and adds the parts written in Python: floating-point error state,
``flatiter``, ``ndarray.flags``, and the ``linalg``, ``fft``, ``random``, ``testing`` and
``exceptions`` submodules. The behavior targets NumPy 2.5.
"""

from _numpy import *
from _numpy_reduce import *
from _numpy_shape import *
from _numpy_sort import *
from _numpy_products import *
from _numpy_math import *
from _numpy_io import *
from _numpy_print import *
from numpy._errstate import errstate, geterr, seterr
from numpy._numeric import allclose, array_equal, array_equiv, isclose
from numpy._methods import average, count_nonzero, mean, ptp, std, var
from numpy._nanfunctions import (
    nanargmax,
    nanargmin,
    nancumprod,
    nancumsum,
    nanmax,
    nanmean,
    nanmin,
    nanprod,
    nanstd,
    nansum,
    nanvar,
)

__version__ = "2.5.3"

newaxis = None


class flatiter:
    """``ndarray.flat``: a C-order, one-dimensional view of an array's elements."""

    def __init__(self, base):
        self.base = base

    def __len__(self):
        return self.base.size

    def __iter__(self):
        return iter(ravel(self.base))

    def _positions(self, index):
        return arange(self.base.size).reshape(self.base.shape).ravel()[index]

    def __getitem__(self, index):
        return ravel(self.base)[index]

    def __setitem__(self, index, value):
        positions = self._positions(index)
        put(self.base, positions, value)

    def __array__(self, dtype=None, copy=None):
        return ravel(self.base).copy()


class _flagsobj:
    """``ndarray.flags``: memory-layout flags. Setting ``writeable`` updates the array."""

    def __init__(self, array):
        object.__setattr__(self, "_array", array)

    @property
    def writeable(self):
        return _writeable(self._array)

    @writeable.setter
    def writeable(self, value):
        self._array.setflags(write=value)

    @property
    def c_contiguous(self):
        return _c_contiguous(self._array)

    @property
    def f_contiguous(self):
        array = self._array
        return array.ndim <= 1 and _c_contiguous(array) or _c_contiguous(array.T)

    @property
    def owndata(self):
        return self._array.base is None

    @property
    def aligned(self):
        return True

    def __getitem__(self, key):
        names = {
            "WRITEABLE": "writeable",
            "W": "writeable",
            "C_CONTIGUOUS": "c_contiguous",
            "C": "c_contiguous",
            "F_CONTIGUOUS": "f_contiguous",
            "F": "f_contiguous",
            "OWNDATA": "owndata",
            "O": "owndata",
            "ALIGNED": "aligned",
            "A": "aligned",
        }
        if key not in names:
            raise KeyError(f"Unknown flag {key}")
        return getattr(self, names[key])

    def __repr__(self):
        return "\n".join(
            f"  {name} : {getattr(self, attribute)}"
            for name, attribute in (
                ("C_CONTIGUOUS", "c_contiguous"),
                ("F_CONTIGUOUS", "f_contiguous"),
                ("OWNDATA", "owndata"),
                ("WRITEABLE", "writeable"),
                ("ALIGNED", "aligned"),
            )
        )


from _numpy import _c_contiguous, _writeable

import numpy.exceptions as exceptions
import numpy.linalg as linalg
import numpy.fft as fft
import numpy.random as random
import numpy.testing as testing
