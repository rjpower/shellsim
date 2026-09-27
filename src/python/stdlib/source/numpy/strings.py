"""`numpy.strings`: elementwise string operations over `str` (and `object`-of-`str`) arrays.

The native `_numpy_strings` module (see `src/python/stdlib/numpy/strings.rs`) exports no
functions of its own; everything here is built from ordinary `str` methods applied
element-by-element, broadcasting extra array arguments the way a ufunc would. Packing the
Python results back through `np.array(...)` gives the result its dtype for free: a `str`
result array gets the width of its longest element, a comparison gets `bool`, and a count or
index gets `int64`, exactly as constructing an array from those Python values normally would.

Functions that take a per-element parameter other than the input arrays (`width`, `fillchar`,
`start`, `end`, separators, `mod`'s `values`, ...) apply that parameter uniformly to every
element rather than broadcasting it too; NumPy allows array-valued parameters there, which
shellsim does not reproduce.

NumPy sizes a result array's dtype from an internal worst-case buffer estimate for the
operation (e.g. `strip` keeps its input's width even when every stripped string is shorter);
shellsim instead sizes it from the actual output text, as plain `np.array(...)` construction
would. Values always match; the reported `.dtype.itemsize` can be narrower.

`decode` and `encode` are absent because they convert to and from the `bytes_` (`S`) dtype,
which shellsim's NumPy does not implement (see the "Unsupported frontier" section of
docs/numpy.md).
"""

import numpy as np

_SLICE_UNSET = object()


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
    # NumPy broadcasts `values` against `a` (one substitution per element); shellsim applies
    # it uniformly instead, per the module-level note above.
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


def _partition3(a, method):
    # `partition`/`rpartition` return three same-shaped arrays (before, separator, after),
    # not one array of 3-tuples: matches `numpy.strings`, which reports each part separately.
    array = _as_element_array(a)
    flat = [method(value) for value in array.reshape(-1).tolist()]
    columns = zip(*flat) if flat else ((), (), ())
    return tuple(np.array(column).reshape(array.shape) for column in columns)


def partition(a, sep):
    return _partition3(a, lambda s: s.partition(sep))


def rpartition(a, sep):
    return _partition3(a, lambda s: s.rpartition(sep))


def slice(a, start=None, stop=_SLICE_UNSET, step=None):
    """Slice every element with `s[start:stop:step]`.

    Matches the real `numpy.strings.slice`'s single-argument shorthand: a lone positional
    argument is `stop` (as in the builtin `slice(stop)`), not `start`.
    """
    if stop is _SLICE_UNSET:
        start, stop = None, start
    return _map1(lambda s: s[start:stop:step], a)


def translate(a, table, deletechars=None):
    # NumPy's own `str`-dtype `translate` leaves `deletechars` without effect (confirmed
    # against NumPy 2.5.3), so shellsim reproduces exactly `str.translate(table)`.
    return _map1(lambda s: s.translate(table), a)


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
