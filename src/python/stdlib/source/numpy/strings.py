"""`numpy.strings`: elementwise string operations over `str` (and `object`-of-`str`) arrays.

The native `_numpy_strings` module (see `src/python/stdlib/numpy/strings.rs`) exports no
functions of its own; everything here is built from ordinary `str` methods applied
element-by-element, broadcasting extra array arguments the way a ufunc would. Packing the
Python results back through `np.array(...)` gives the result its dtype for free: a `str`
result array gets the width of its longest element, a comparison gets `bool`, and a count or
index gets `int64`, exactly as constructing an array from those Python values normally would.

Functions that take a per-element parameter other than the input arrays (`width`, `fillchar`,
`start`, `end`, separators, ...) apply that parameter uniformly to every element rather than
broadcasting it too; NumPy allows array-valued parameters there, which shellsim does not
reproduce.
"""

import numpy as np


def _as_element_array(value):
    return value if isinstance(value, np.ndarray) else np.array(value)


def _map1(func, a):
    array = _as_element_array(a)
    flat = [func(value) for value in array.reshape(-1).tolist()]
    return np.array(flat).reshape(array.shape)


def _map2(func, a, b):
    a = _as_element_array(a)
    b = _as_element_array(b)
    a, b = np.broadcast_arrays(a, b)
    flat_a = a.reshape(-1).tolist()
    flat_b = b.reshape(-1).tolist()
    return np.array([func(x, y) for x, y in zip(flat_a, flat_b)]).reshape(a.shape)


# -- concatenation and repetition -------------------------------------------


def add(x1, x2):
    return _map2(lambda a, b: a + b, x1, x2)


def multiply(a, i):
    return _map2(lambda s, n: s * n, a, i)


def mod(a, values):
    return _map1(lambda s: s % values, a)


# -- case ---------------------------------------------------------------


def capitalize(a):
    return _map1(str.capitalize, a)


def lower(a):
    return _map1(str.lower, a)


def upper(a):
    return _map1(str.upper, a)


def swapcase(a):
    return _map1(str.swapcase, a)


def title(a):
    return _map1(str.title, a)


# -- padding --------------------------------------------------------------


def center(a, width, fillchar=" "):
    return _map1(lambda s: s.center(width, fillchar), a)


def ljust(a, width, fillchar=" "):
    return _map1(lambda s: s.ljust(width, fillchar), a)


def rjust(a, width, fillchar=" "):
    return _map1(lambda s: s.rjust(width, fillchar), a)


def zfill(a, width):
    return _map1(lambda s: s.zfill(width), a)


def expandtabs(a, tabsize=8):
    return _map1(lambda s: s.expandtabs(tabsize), a)


# -- trimming ---------------------------------------------------------------


def strip(a, chars=None):
    return _map1(lambda s: s.strip(chars), a)


def lstrip(a, chars=None):
    return _map1(lambda s: s.lstrip(chars), a)


def rstrip(a, chars=None):
    return _map1(lambda s: s.rstrip(chars), a)


# -- search and replace ----------------------------------------------------


def count(a, sub, start=0, end=None):
    return _map1(lambda s: s.count(sub, start, end), a)


def find(a, sub, start=0, end=None):
    return _map1(lambda s: s.find(sub, start, end), a)


def rfind(a, sub, start=0, end=None):
    return _map1(lambda s: s.rfind(sub, start, end), a)


def index(a, sub, start=0, end=None):
    return _map1(lambda s: s.index(sub, start, end), a)


def rindex(a, sub, start=0, end=None):
    return _map1(lambda s: s.rindex(sub, start, end), a)


def replace(a, old, new, count=-1):
    return _map1(lambda s: s.replace(old, new, count), a)


def partition(a, sep):
    return _map1(lambda s: s.partition(sep), a)


def rpartition(a, sep):
    return _map1(lambda s: s.rpartition(sep), a)


def join(sep, seq):
    return _map2(lambda s, parts: s.join(parts), sep, seq)


def split(a, sep=None, maxsplit=-1):
    return _map1(lambda s: s.split(sep, maxsplit), a)


def rsplit(a, sep=None, maxsplit=-1):
    return _map1(lambda s: s.rsplit(sep, maxsplit), a)


def splitlines(a, keepends=False):
    return _map1(lambda s: s.splitlines(keepends), a)


def startswith(a, prefix, start=0, end=None):
    return _map1(lambda s: s.startswith(prefix, start, end), a)


def endswith(a, suffix, start=0, end=None):
    return _map1(lambda s: s.endswith(suffix, start, end), a)


# -- predicates -------------------------------------------------------------


def str_len(a):
    return _map1(len, a)


def isalpha(a):
    return _map1(str.isalpha, a)


def isdigit(a):
    return _map1(str.isdigit, a)


def isdecimal(a):
    return _map1(str.isdecimal, a)


def isnumeric(a):
    return _map1(str.isnumeric, a)


def isalnum(a):
    return _map1(str.isalnum, a)


def isspace(a):
    return _map1(str.isspace, a)


def islower(a):
    return _map1(str.islower, a)


def isupper(a):
    return _map1(str.isupper, a)


def istitle(a):
    return _map1(str.istitle, a)


# -- comparisons (mirrors of the operators, for parity with numpy.strings) --


def equal(x1, x2):
    return _map2(lambda a, b: a == b, x1, x2)


def not_equal(x1, x2):
    return _map2(lambda a, b: a != b, x1, x2)


def greater(x1, x2):
    return _map2(lambda a, b: a > b, x1, x2)


def greater_equal(x1, x2):
    return _map2(lambda a, b: a >= b, x1, x2)


def less(x1, x2):
    return _map2(lambda a, b: a < b, x1, x2)


def less_equal(x1, x2):
    return _map2(lambda a, b: a <= b, x1, x2)
