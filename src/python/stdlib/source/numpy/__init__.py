"""shellsim's NumPy work-alike, targeting NumPy 2.5.3's observable behavior (see ``docs/numpy.md``).

This package star-imports the native modules (arrays, dtypes, ufuncs, and the array-shaped
areas: reductions, shape operations, sorting, products, math kernels, raw-buffer I/O, and
printing digits) and layers the parts NumPy itself writes in Python on top: error/warning types,
``errstate``, ``finfo``/``iinfo``, index-construction helpers, shape composites (``stack``,
``split``, ``kron``, ``block``, ...), set operations, padding, general numeric helpers,
statistics, ``nan*`` reductions, histograms, and ``vectorize``. ``_arrayprint``, ``strings`` and
``linalg`` (owned by other areas of this port) import eagerly, at the end, once every name they
need (from the native modules and the areas above) is already bound here. ``numpy.fft``,
``numpy.random``, ``numpy.testing``, ``numpy.lib``, and the file I/O functions (``save``,
``load``, ...) load on first access through ``__getattr__``, as NumPy does.

``ndarray.flags``, ``generic.flags`` and ``ndarray.flat`` are native getters that call back into
this module by name (``numpy._flagsobj``, ``numpy._scalar_flags``, ``numpy.flatiter``), so those
three classes live here rather than in a submodule.
"""

#: The real builtin, captured before ``from _numpy import *`` rebinds the name ``bool`` in this
#: module's globals to NumPy's scalar type (as real NumPy itself does with ``np.bool``). Every
#: later use of ``bool(...)`` in this file goes through this alias instead.
_python_bool = bool

from _numpy import *
from _numpy import (
    _AxisError,
    _c_contiguous,
    _ComplexWarning,
    _DTypePromotionError,
    _LinAlgError,
    _UFuncTypeError,
    _writeable,
)
from _numpy_io import *
from _numpy_math import *
from _numpy_products import *
from _numpy_reduce import *
from _numpy_shape import *
from _numpy_shape import _normalize_axis_index
from _numpy_sort import *

from numpy._arraypad import *
from numpy._arraysetops import *
from numpy._errstate import *
from numpy._function_base import *
from numpy._getlimits import *
from numpy._histograms import *
from numpy._index_tricks import *
from numpy._nanfunctions import *
from numpy._numeric import *
from numpy._shape_base import *
from numpy._statistics import *
from numpy._vectorize import *
from numpy.exceptions import *

__version__ = "2.5.3"

#: ``a[..., np.newaxis]`` inserts a length-1 axis; NumPy spells this with ``None`` itself.
newaxis = None

#: Character codes grouped by kind, restricted to the dtypes shellsim implements (no
#: ``longdouble``, ``datetime64`` or ``timedelta64``).
typecodes = {
    "Character": "U",
    "Integer": "bhil",
    "UnsignedInteger": "BHIL",
    "Float": "efd",
    "Complex": "FD",
    "AllInteger": "bBhHiIlL",
    "AllFloat": "efdFD",
    "All": "?bhilBHILefdFDUO",
}


class _flagsobj:
    """``ndarray.flags``: contiguity and mutability, by key (``flags["C_CONTIGUOUS"]``, and
    NumPy's short and alias forms) or by lowercase attribute (``flags.c_contiguous``). Each named
    flag is a property.
    """

    _KEYS = ("C_CONTIGUOUS", "F_CONTIGUOUS", "OWNDATA", "WRITEABLE", "ALIGNED", "WRITEBACKIFCOPY")
    _ALIASES = {
        "C": "C_CONTIGUOUS",
        "CONTIGUOUS": "C_CONTIGUOUS",
        "F": "F_CONTIGUOUS",
        "FORTRAN": "F_CONTIGUOUS",
        "O": "OWNDATA",
        "W": "WRITEABLE",
        "A": "ALIGNED",
        "X": "WRITEBACKIFCOPY",
    }

    def __init__(self, array):
        self._array = array

    def _flag(self, key):
        if key == "C_CONTIGUOUS":
            return _python_bool(_c_contiguous(self._array))
        if key == "F_CONTIGUOUS":
            return _is_f_contiguous(self._array)
        if key == "OWNDATA":
            return self._array.base is None
        if key == "WRITEABLE":
            return _python_bool(_writeable(self._array))
        if key == "ALIGNED":
            return True
        if key == "WRITEBACKIFCOPY":
            return False
        raise KeyError(f"Unknown flag: {key}")

    def _resolve(self, key):
        if key == "FORC":
            return self._flag("C_CONTIGUOUS") or self._flag("F_CONTIGUOUS")
        if key == "FNC":
            return self._flag("F_CONTIGUOUS") and not self._flag("C_CONTIGUOUS")
        if key in self._KEYS:
            return self._flag(key)
        return self._flag(self._ALIASES[key])

    def __getitem__(self, key):
        try:
            return self._resolve(key)
        except KeyError:
            raise KeyError(f"Unknown flag: {key}") from None

    def __setitem__(self, key, value):
        if key in ("WRITEABLE", "W"):
            self.writeable = value
            return
        raise ValueError(f"cannot set flag {key!r}")

    @property
    def c_contiguous(self):
        return self._flag("C_CONTIGUOUS")

    @property
    def f_contiguous(self):
        return self._flag("F_CONTIGUOUS")

    #: NumPy's other spelling for ``c_contiguous``.
    contiguous = c_contiguous

    #: NumPy's other spelling for ``f_contiguous``.
    fortran = f_contiguous

    @property
    def owndata(self):
        return self._flag("OWNDATA")

    @property
    def writeable(self):
        return self._flag("WRITEABLE")

    @writeable.setter
    def writeable(self, value):
        self._array.setflags(write=_python_bool(value))

    @property
    def aligned(self):
        return self._flag("ALIGNED")

    @property
    def writebackifcopy(self):
        return self._flag("WRITEBACKIFCOPY")

    @property
    def forc(self):
        return self._resolve("FORC")

    @property
    def fnc(self):
        return self._resolve("FNC")

    def __repr__(self):
        return "".join(f"  {key} : {self._flag(key)}\n" for key in self._KEYS)


def _is_f_contiguous(array):
    """Whether `array` is stored Fortran-contiguous: strides grow from the first axis outward."""
    expected = array.dtype.itemsize
    for size, stride in zip(array.shape, array.strides):
        if size == 1:
            continue
        if stride != expected:
            return False
        expected *= size
    return True


class _scalar_flags:
    """``generic.flags``: a NumPy scalar is always contiguous, owns its one element, and is
    read-only."""

    def __init__(self, value):
        self._value = value

    def _flag(self, key):
        if key in ("C_CONTIGUOUS", "F_CONTIGUOUS", "OWNDATA", "ALIGNED"):
            return True
        if key in ("WRITEABLE", "WRITEBACKIFCOPY"):
            return False
        raise KeyError(f"Unknown flag: {key}")

    def __getitem__(self, key):
        key = _flagsobj._ALIASES.get(key, key)
        try:
            return self._flag(key)
        except KeyError:
            raise KeyError(f"Unknown flag: {key}") from None

    @property
    def c_contiguous(self):
        return True

    #: A scalar is always both C- and Fortran-contiguous.
    f_contiguous = c_contiguous
    contiguous = c_contiguous
    fortran = c_contiguous

    @property
    def owndata(self):
        return True

    @property
    def aligned(self):
        return True

    @property
    def writeable(self):
        return False

    @property
    def writebackifcopy(self):
        return False

    @property
    def forc(self):
        return True

    @property
    def fnc(self):
        return False

    def __repr__(self):
        return "".join(f"  {key} : {self._flag(key)}\n" for key in _flagsobj._KEYS)


class flatiter:
    """``ndarray.flat``: a C-order iterator and indexer over `array`'s logical elements.

    Every read and write goes through `array` itself via :func:`numpy.unravel_index`, so it
    follows the array's own view (strides included) rather than a raveled copy: writes reach
    the original storage even when `array` is not contiguous.
    """

    def __init__(self, array):
        self._array = array

    def __len__(self):
        return self._array.size

    def __iter__(self):
        shape = self._array.shape
        for position in range(self._array.size):
            yield self._array[unravel_index(position, shape)]

    def _positions(self, key):
        size = self._array.size
        if isinstance(key, slice):
            return list(range(*key.indices(size)))
        if isinstance(key, (list, tuple, ndarray)):
            return [int(i) % size if int(i) < 0 else int(i) for i in asanyarray(key).reshape(-1).tolist()]
        index = int(key)
        return index % size if index < 0 else index

    def __getitem__(self, key):
        shape = self._array.shape
        positions = self._positions(key)
        if isinstance(positions, int):
            return self._array[unravel_index(positions, shape)]
        return array([self._array[unravel_index(position, shape)] for position in positions])

    def __setitem__(self, key, value):
        shape = self._array.shape
        positions = self._positions(key)
        if isinstance(positions, int):
            self._array[unravel_index(positions, shape)] = value
            return
        positions = list(positions)
        values = [value] * len(positions) if isscalar(value) else broadcast_to(asanyarray(value), (len(positions),))
        for position, one_value in zip(positions, values):
            self._array[unravel_index(position, shape)] = one_value


_LAZY_SUBMODULES = ("fft", "random", "testing", "lib")
_LAZY_IO_FUNCTIONS = ("save", "load", "savez", "savez_compressed", "savetxt", "loadtxt", "genfromtxt")


def __getattr__(name):
    # shellsim's interpreter has no `globals()` builtin, so this cannot cache the resolved
    # value back into the module's own namespace the way CPython's lazy-import recipes do.
    # `__import__` on an already-imported module is cheap (it returns the cached module
    # object rather than re-running it), so re-resolving on every access is still fine.
    if name in _LAZY_SUBMODULES:
        return __import__(f"numpy.{name}", fromlist=[name])
    if name in _LAZY_IO_FUNCTIONS:
        npyio = __import__("numpy.lib.npyio", fromlist=["npyio"])
        return getattr(npyio, name)
    raise AttributeError(f"module 'numpy' has no attribute {name!r}")


# Eager, and last: these modules do `from numpy import (...)` (or `np.X`) at their own top
# level or on every call, which needs every name above already bound here.
from numpy._arrayprint import *

import numpy.strings as strings
import numpy.linalg as linalg
