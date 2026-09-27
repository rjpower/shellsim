"""`numpy.strings`: elementwise string operations over `str` (and `object`-of-`str`) arrays.

The native `_numpy_strings` module (see `src/python/stdlib/numpy/strings.rs`) exports no
functions of its own; everything here is built from ordinary `str` methods applied
element-by-element. Every argument -- the input array and any per-element parameter such as
`width`, `fillchar`, `start`, `end`, `chars`, a separator, `old`/`new`/`count`, `tabsize`, or
`mod`'s `values` -- is broadcast together first, the way a ufunc broadcasts its operands, so an
array-valued parameter supplies one value per element exactly as NumPy's own `numpy.strings`
does. `translate`'s `table` is the one exception: NumPy documents it as a single shared
translation table for the whole call, not an array_like parameter, so shellsim does not
broadcast it either.

A `str`-dtype result's width is chosen to match NumPy's own rule wherever black-box probing
against NumPy 2.5.3 pinned one down:

- `capitalize`, `lower`, `upper`, `swapcase`, `title`, `strip`, `lstrip`, `rstrip`, `slice`, and
  `translate` keep the input's width exactly, even when every result is shorter -- and, matching
  NumPy's own fixed-width buffers, a result that comes out *longer* than the input (case folding
  can grow a string, e.g. `"straße".upper()`) is truncated to fit rather than widening the
  array.
- `center`, `ljust`, `rjust`, and `zfill` grow to `max(input width, requested width)`.
- `add`, `multiply`, `replace`, `mod`, and `partition`/`rpartition` size the result from the
  actual computed text (like plain `np.array(...)` construction), which is what NumPy itself
  does for these too, except in one corner case: an all-empty `partition`/`rpartition` column
  across the *whole* array reports `itemsize` 0 in NumPy but 1 in shellsim (`np.array`'s own
  floor for an all-empty string array), since NumPy allocates that column from a rule internal
  to its C implementation rather than from the text it wrote.
- `expandtabs` could not be pinned down: probing it against varying tab sizes and input text
  shows NumPy allocating more than the expanded text needs, by an amount that is not a simple
  function of the input width, the tab size, or the number of tab characters present. shellsim
  sizes its result from the actual expanded text instead. Values always match; only
  `expandtabs`'s reported `.dtype.itemsize` can differ from NumPy's.

`decode` and `encode` are absent because they convert to and from the `bytes_` (`S`) dtype,
which shellsim's NumPy does not implement (see the "Unsupported frontier" section of
docs/numpy.md).
"""

import numpy as np

_SLICE_UNSET = object()


def _as_element_array(value):
    return value if isinstance(value, np.ndarray) else np.array(value)


def _broadcast(*args):
    return np.broadcast_arrays(*(_as_element_array(value) for value in args))


def _map(func, *args):
    """Broadcast every argument together and apply `func` elementwise, like a ufunc.

    The result's dtype is inferred from the returned Python values, exactly as `np.array(...)`
    would: right for `bool`/`int` results (comparisons, `count`, `find`, the `isX` predicates),
    and for the handful of `str` results (`add`, `multiply`, `replace`, `mod`, `expandtabs`)
    whose width NumPy itself computes from the actual output text.
    """
    arrays = _broadcast(*args)
    shape = arrays[0].shape
    flats = [array.reshape(-1).tolist() for array in arrays]
    results = [func(*values) for values in zip(*flats)]
    return np.array(results).reshape(shape)


def _char_width(dtype):
    # shellsim's `<U>` dtypes report `itemsize` in bytes, 4 per character (see
    # `numpy/lib/format.py`'s `_dtype_pickle_fields`, which relies on the same convention).
    return max(dtype.itemsize // 4, 1)


def _fixed_width_map(func, args, width):
    arrays = _broadcast(*args)
    shape = arrays[0].shape
    flats = [array.reshape(-1).tolist() for array in arrays]
    results = [func(*values) for values in zip(*flats)]
    return np.array(results, dtype=f"<U{width}").reshape(shape)


def _preserve(func, a, *rest):
    """Elementwise map whose `str`-dtype result keeps `a`'s width exactly (NumPy's rule for
    case folding, stripping, slicing, and translation). Object arrays have no fixed width to
    preserve -- NumPy's own `numpy.strings` does not accept them either -- so they fall back to
    `_map`'s content-based sizing, which is what this module already did for them before
    `str`-dtype width fidelity was added."""
    array = _as_element_array(a)
    if array.dtype.kind != "U":
        return _map(func, array, *rest)
    return _fixed_width_map(func, (array, *rest), _char_width(array.dtype))


def _grow(func, a, width_param, *rest):
    """Elementwise map whose `str`-dtype result grows to `max(a`'s width, `width_param`'s
    largest requested value)`, NumPy's rule for `center`/`ljust`/`rjust`/`zfill`."""
    array = _as_element_array(a)
    width_array = _as_element_array(width_param)
    if array.dtype.kind != "U":
        return _map(func, array, width_array, *rest)
    requested = int(width_array.max()) if width_array.size else 0
    width = max(_char_width(array.dtype), requested, 1)
    return _fixed_width_map(func, (array, width_array, *rest), width)


# -- concatenation and repetition -------------------------------------------


def add(x1, x2):
    return _map(lambda a, b: a + b, x1, x2)


def multiply(a, i):
    return _map(lambda s, n: s * n, a, i)


def mod(a, values):
    return _map(lambda s, v: s % v, a, values)


# -- case ---------------------------------------------------------------


def capitalize(a):
    return _preserve(str.capitalize, a)


def lower(a):
    return _preserve(str.lower, a)


def upper(a):
    return _preserve(str.upper, a)


def swapcase(a):
    return _preserve(str.swapcase, a)


def title(a):
    return _preserve(str.title, a)


# -- padding --------------------------------------------------------------


def center(a, width, fillchar=" "):
    return _grow(lambda s, w, f: s.center(w, f), a, width, fillchar)


def ljust(a, width, fillchar=" "):
    return _grow(lambda s, w, f: s.ljust(w, f), a, width, fillchar)


def rjust(a, width, fillchar=" "):
    return _grow(lambda s, w, f: s.rjust(w, f), a, width, fillchar)


def zfill(a, width):
    return _grow(lambda s, w: s.zfill(w), a, width)


def expandtabs(a, tabsize=8):
    # See the module docstring: NumPy's own width here is not a pinnable function of the
    # input width, tabsize, or tab count, so this keeps `_map`'s content-based sizing.
    return _map(lambda s, t: s.expandtabs(t), a, tabsize)


# -- trimming ---------------------------------------------------------------


def strip(a, chars=None):
    return _preserve(lambda s, c: s.strip(c), a, chars)


def lstrip(a, chars=None):
    return _preserve(lambda s, c: s.lstrip(c), a, chars)


def rstrip(a, chars=None):
    return _preserve(lambda s, c: s.rstrip(c), a, chars)


# -- search and replace ----------------------------------------------------


def count(a, sub, start=0, end=None):
    return _map(lambda s, u, i, j: s.count(u, i, j), a, sub, start, end)


def find(a, sub, start=0, end=None):
    return _map(lambda s, u, i, j: s.find(u, i, j), a, sub, start, end)


def rfind(a, sub, start=0, end=None):
    return _map(lambda s, u, i, j: s.rfind(u, i, j), a, sub, start, end)


def index(a, sub, start=0, end=None):
    return _map(lambda s, u, i, j: s.index(u, i, j), a, sub, start, end)


def rindex(a, sub, start=0, end=None):
    return _map(lambda s, u, i, j: s.rindex(u, i, j), a, sub, start, end)


def replace(a, old, new, count=-1):
    return _map(lambda s, o, n, c: s.replace(o, n, c), a, old, new, count)


def _partition3(a, sep, method_name):
    # `partition`/`rpartition` return three same-shaped arrays (before, separator, after),
    # not one array of 3-tuples: matches `numpy.strings`, which reports each part separately.
    arrays = _broadcast(a, sep)
    shape = arrays[0].shape
    flats = [array.reshape(-1).tolist() for array in arrays]
    results = [getattr(s, method_name)(part) for s, part in zip(*flats)]
    columns = zip(*results) if results else ((), (), ())
    return tuple(np.array(column).reshape(shape) for column in columns)


def partition(a, sep):
    return _partition3(a, sep, "partition")


def rpartition(a, sep):
    return _partition3(a, sep, "rpartition")


def slice(a, start=None, stop=_SLICE_UNSET, step=None):
    """Slice every element with `s[start:stop:step]`.

    Matches the real `numpy.strings.slice`'s single-argument shorthand: a lone positional
    argument is `stop` (as in the builtin `slice(stop)`), not `start`.
    """
    if stop is _SLICE_UNSET:
        start, stop = None, start
    return _preserve(lambda s, st, sp, sk: s[st:sp:sk], a, start, stop, step)


def translate(a, table, deletechars=None):
    # NumPy's own `str`-dtype `translate` leaves `deletechars` without effect (confirmed
    # against NumPy 2.5.3), so shellsim reproduces exactly `str.translate(table)`. `table`
    # itself is not broadcast; see the module docstring.
    return _preserve(lambda s: s.translate(table), a)


def startswith(a, prefix, start=0, end=None):
    return _map(lambda s, p, i, j: s.startswith(p, i, j), a, prefix, start, end)


def endswith(a, suffix, start=0, end=None):
    return _map(lambda s, x, i, j: s.endswith(x, i, j), a, suffix, start, end)


# -- predicates -------------------------------------------------------------


def str_len(a):
    return _map(len, a)


def isalpha(a):
    return _map(str.isalpha, a)


def isdigit(a):
    return _map(str.isdigit, a)


def isdecimal(a):
    return _map(str.isdecimal, a)


def isnumeric(a):
    return _map(str.isnumeric, a)


def isalnum(a):
    return _map(str.isalnum, a)


def isspace(a):
    return _map(str.isspace, a)


def islower(a):
    return _map(str.islower, a)


def isupper(a):
    return _map(str.isupper, a)


def istitle(a):
    return _map(str.istitle, a)


# -- comparisons (mirrors of the operators, for parity with numpy.strings) --


def equal(x1, x2):
    return _map(lambda a, b: a == b, x1, x2)


def not_equal(x1, x2):
    return _map(lambda a, b: a != b, x1, x2)


def greater(x1, x2):
    return _map(lambda a, b: a > b, x1, x2)


def greater_equal(x1, x2):
    return _map(lambda a, b: a >= b, x1, x2)


def less(x1, x2):
    return _map(lambda a, b: a < b, x1, x2)


def less_equal(x1, x2):
    return _map(lambda a, b: a <= b, x1, x2)
