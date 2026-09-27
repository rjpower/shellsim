"""shellsim's NumPy.

Arrays, dtypes, ufuncs and most functions are native and live in the ``_numpy*`` modules; this
package re-exports them and adds the parts written in Python: floating-point error state,
``flatiter``, ``ndarray.flags``, order statistics, histograms, set operations, file I/O, and
the ``linalg``, ``fft``, ``random``, ``strings``, ``testing``, ``lib`` and ``exceptions``
submodules. The behavior targets NumPy 2.5.
"""

from _numpy import *
from _numpy_reduce import *
from _numpy_shape import *
from _numpy_sort import *
from _numpy_products import *
from _numpy_math import *
from _numpy_io import *
from numpy._arrayprint import (
    array2string,
    array_repr,
    array_str,
    format_float_positional,
    format_float_scientific,
    get_printoptions,
    printoptions,
    set_printoptions,
)
from numpy._errstate import errstate, geterr, geterrcall, seterr, seterrcall
from numpy._getlimits import finfo, iinfo
from numpy._numeric import allclose, array_equal, array_equiv, astype, isclose, isdtype
from numpy._shape_base import (
    append,
    apply_along_axis,
    apply_over_axes,
    array_split,
    atleast_1d,
    atleast_2d,
    atleast_3d,
    column_stack,
    diff,
    dsplit,
    dstack,
    fliplr,
    flipud,
    hsplit,
    hstack,
    put_along_axis,
    resize,
    roll,
    rot90,
    split,
    stack,
    take_along_axis,
    tile,
    tri,
    tril,
    triu,
    vander,
    vsplit,
    vstack,
)
from numpy._function_base import (
    angle,
    argwhere,
    around,
    clip,
    convolve,
    correlate,
    cross,
    cumulative_prod,
    cumulative_sum,
    extract,
    fix,
    flatnonzero,
    gradient,
    imag,
    interp,
    iscomplex,
    iscomplexobj,
    isneginf,
    isposinf,
    isreal,
    isrealobj,
    iterable,
    mintypecode,
    nan_to_num,
    place,
    polyfit,
    putmask,
    polyval,
    real,
    real_if_close,
    round,
    select,
    vecdot,
)
from numpy._vectorize import typecodes, vectorize
from numpy._methods import average, count_nonzero, mean, ptp, std, var
from numpy._statistics import corrcoef, cov, median, percentile, quantile
from numpy._histograms import digitize, histogram, histogram_bin_edges
from numpy._index_tricks import index_exp, ix_, ndenumerate, ndindex, s_
from numpy._arraysetops import (
    ediff1d,
    intersect1d,
    isin,
    setdiff1d,
    setxor1d,
    union1d,
    unique,
    unique_all,
    unique_counts,
    unique_inverse,
    unique_values,
)
from numpy._nanfunctions import (
    nanargmax,
    nanargmin,
    nancumprod,
    nancumsum,
    nanmax,
    nanmean,
    nanmedian,
    nanmin,
    nanpercentile,
    nanprod,
    nanquantile,
    nanstd,
    nansum,
    nanvar,
)

__version__ = "2.5.3"

newaxis = None
amax = max
amin = min
concat = concatenate


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
        self._array = array

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

    @property
    def fnc(self):
        return self.f_contiguous and not self.c_contiguous

    @property
    def forc(self):
        return self.f_contiguous or self.c_contiguous

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
from numpy._arraypad import pad


_NPYIO_EXPORTS = {
    "genfromtxt",
    "load",
    "loadtxt",
    "save",
    "savetxt",
    "savez",
    "savez_compressed",
}


def __getattr__(attr):
    # As in NumPy, the larger submodules load on first access rather than with the package.
    if attr == "fft":
        import numpy.fft as fft

        return fft
    elif attr == "random":
        import numpy.random as random

        return random
    elif attr == "strings":
        import numpy.strings as strings

        return strings
    elif attr == "testing":
        import numpy.testing as testing

        return testing
    elif attr == "lib":
        import numpy.lib as lib

        return lib
    elif attr in _NPYIO_EXPORTS:
        from numpy.lib import npyio

        return getattr(npyio, attr)
    raise AttributeError(f"module {__name__!r} has no attribute {attr!r}")
