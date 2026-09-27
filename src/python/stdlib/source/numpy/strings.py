"""``numpy.strings``: element-wise operations on ``str`` arrays.

NumPy implements most of these as string ufuncs in C. Here each function applies the matching
Python ``str`` method to every broadcast element and sizes the result dtype by NumPy's rules in
``numpy/_core/strings.py``: case changes and stripping keep the input's width, padding widens
to the requested width, and ``multiply`` and ``replace`` compute the widest result. Results
from functions NumPy implements as plain ufuncs are scalars for scalar input; functions that
NumPy runs with an ``out=`` buffer or through ``_vec_string`` return 0-d arrays.

Byte-string (``S``) arrays do not exist here, so ``encode`` and ``decode`` are not provided.
"""

import numpy as np
from numpy import (
    add,
    equal,
    greater,
    greater_equal,
    less,
    less_equal,
    not_equal,
)

__all__ = [
    "equal",
    "not_equal",
    "less",
    "less_equal",
    "greater",
    "greater_equal",
    "add",
    "multiply",
    "isalpha",
    "isdigit",
    "isspace",
    "isalnum",
    "islower",
    "isupper",
    "istitle",
    "isdecimal",
    "isnumeric",
    "str_len",
    "find",
    "rfind",
    "index",
    "rindex",
    "count",
    "startswith",
    "endswith",
    "lstrip",
    "rstrip",
    "strip",
    "replace",
    "expandtabs",
    "center",
    "ljust",
    "rjust",
    "zfill",
    "partition",
    "rpartition",
    "slice",
    "upper",
    "lower",
    "swapcase",
    "capitalize",
    "title",
    "mod",
    "translate",
]

MAX = np.iinfo(np.int64).max
_NoValue = object()


def _dtype_class(dtype):
    name = dtype.name
    if dtype.kind == "U":
        return "StrDType"
    if dtype.kind == "O":
        return "ObjectDType"
    if dtype.kind == "b":
        return "BoolDType"
    return name[0].upper() + name[1:] + "DType"


def _string_array(a, ufunc):
    """``a`` as a ``str`` array, with the error NumPy's string ufunc ``ufunc`` raises."""
    a = np.asanyarray(a)
    if a.dtype.kind != "U":
        raise TypeError(
            f"ufunc '{ufunc}' did not contain a loop with signature matching types "
            f"<class 'numpy.dtypes.{_dtype_class(a.dtype)}'> -> None"
        )
    return a


def _vec_string_array(a):
    a = np.asarray(a)
    if a.dtype.kind != "U":
        raise TypeError("string operation on non-string array")
    return a


def _integer_argument(value, name):
    value = np.asanyarray(value)
    if not np.issubdtype(value.dtype, np.integer):
        raise TypeError(f"unsupported type {value.dtype} for operand '{name}'")
    return value


def _apply(function, *operands):
    """Call ``function`` on each broadcast tuple of elements; return the results and shape."""
    arrays = np.broadcast_arrays(*[np.asanyarray(operand) for operand in operands])
    shape = arrays[0].shape
    columns = [array.ravel().tolist() for array in arrays]
    return [function(*items) for items in zip(*columns)], shape


def _values(results, shape, dtype, scalar):
    """An array of ``results``; a 0-d result is a NumPy scalar when ``scalar`` is set."""
    out = np.array(results, dtype=dtype).reshape(shape)
    if scalar and out.ndim == 0:
        return out[()]
    return out


def _strings(results, shape, chars, scalar=False):
    return _values(results, shape, f"U{max(int(chars), 1)}", scalar)


def _chars(a):
    return a.dtype.itemsize // 4


def multiply(a, i):
    a = np.asanyarray(a)
    i = _integer_argument(i, "i")
    i = np.maximum(i, 0)
    a = _string_array(a, "multiply")
    results, shape = _apply(lambda text, times: text * times, a, i)
    widest = max((len(text) for text in results), default=0)
    return _strings(results, shape, widest)


def _predicate(name):
    def predicate(a):
        a = _string_array(a, name)
        results, shape = _apply(lambda text: getattr(text, name)(), a)
        return _values(results, shape, bool, True)

    return predicate


isalpha = _predicate("isalpha")
isdigit = _predicate("isdigit")
isspace = _predicate("isspace")
isalnum = _predicate("isalnum")
islower = _predicate("islower")
isupper = _predicate("isupper")
istitle = _predicate("istitle")
isdecimal = _predicate("isdecimal")
isnumeric = _predicate("isnumeric")


def str_len(a):
    a = _string_array(a, "str_len")
    results, shape = _apply(len, a)
    return _values(results, shape, np.int64, True)


def _search(name, method, missing=None):
    def search(a, sub, start=0, end=None):
        a = _string_array(a, name)
        end = MAX if end is None else end

        def one(text, needle, first, last):
            position = getattr(text, method)(needle, first, last)
            if missing is not None and position < 0:
                raise ValueError(missing)
            return position

        results, shape = _apply(one, a, sub, start, end)
        return _values(results, shape, np.int64, True)

    return search


find = _search("find", "find")
rfind = _search("rfind", "rfind")
index = _search("index", "find", "substring not found")
rindex = _search("rindex", "rfind", "substring not found")
count = _search("count", "count")


def _affix(name):
    def affix(a, prefix, start=0, end=None):
        a = _string_array(a, name)
        end = MAX if end is None else end
        results, shape = _apply(
            lambda text, part, first, last: getattr(text, name)(part, first, last),
            a,
            prefix,
            start,
            end,
        )
        return _values(results, shape, bool, True)

    return affix


startswith = _affix("startswith")
endswith = _affix("endswith")


def _strip(name):
    def strip(a, chars=None):
        a = _string_array(a, name)
        if chars is None:
            results, shape = _apply(lambda text: getattr(text, name)(), a)
        else:
            results, shape = _apply(lambda text, remove: getattr(text, name)(remove), a, chars)
        return _strings(results, shape, _chars(a), scalar=True)

    return strip


lstrip = _strip("lstrip")
rstrip = _strip("rstrip")
strip = _strip("strip")


def _case(name):
    def case(a):
        a = _vec_string_array(a)
        results, shape = _apply(lambda text: getattr(text, name)(), a)
        return _strings(results, shape, _chars(a))

    return case


upper = _case("upper")
lower = _case("lower")
swapcase = _case("swapcase")
capitalize = _case("capitalize")
title = _case("title")


def replace(a, old, new, count=-1):
    count = _integer_argument(count, "count")
    a = _string_array(a, "replace")
    results, shape = _apply(
        lambda text, before, after, limit: text.replace(before, after, limit),
        a,
        old,
        new,
        count,
    )
    widest = max((len(text) for text in results), default=0)
    return _strings(results, shape, widest)


def expandtabs(a, tabsize=8):
    a = _string_array(a, "expandtabs")
    results, shape = _apply(lambda text, size: text.expandtabs(size), a, tabsize)
    widest = max((len(text) for text in results), default=0)
    return _strings(results, shape, widest)


def _justify(name):
    def justify(a, width, fillchar=" "):
        width = _integer_argument(width, "width")
        a = _string_array(a, name)
        fill = np.asanyarray(fillchar)
        if np.any(str_len(fill) != 1):
            raise TypeError("The fill character must be exactly one character long")
        results, shape = _apply(
            lambda text, size, char: getattr(text, name)(size, char), a, width, fill
        )
        widest = np.max(np.maximum(str_len(a), width), initial=0)
        return _strings(results, shape, widest)

    return justify


center = _justify("center")
ljust = _justify("ljust")
rjust = _justify("rjust")


def zfill(a, width):
    width = _integer_argument(width, "width")
    a = _string_array(a, "zfill")
    results, shape = _apply(lambda text, size: text.zfill(size), a, width)
    widest = np.max(np.maximum(str_len(a), width), initial=0)
    return _strings(results, shape, widest)


def _split_at(name, method):
    def split_at(a, sep):
        a = _string_array(a, name)
        results, shape = _apply(lambda text, separator: getattr(text, method)(separator), a, sep)
        parts = []
        for field in range(3):
            texts = [result[field] for result in results]
            parts.append(_strings(texts, shape, _field_chars(texts, field, results, sep)))
        return tuple(parts)

    return split_at


def _field_chars(texts, field, results, sep):
    """NumPy sizes the separator field by the separator even where it is absent, unless it is
    absent everywhere."""
    if field == 1:
        if all(result[1] == "" for result in results):
            return 1
        return np.max(str_len(np.asanyarray(sep)), initial=0)
    return max((len(text) for text in texts), default=0)


partition = _split_at("partition", "partition")
rpartition = _split_at("rpartition", "rpartition")


def slice(a, start=None, stop=_NoValue, step=None, /):
    if stop is _NoValue:
        stop = start
        start = None
    if step is None:
        step = 1
    step = _integer_argument(step, "step")
    if np.any(step == 0):
        raise ValueError("slice step cannot be zero")
    a = _string_array(a, "_slice")
    results, shape = _apply(
        lambda text, first, last, stride: text[first:last:stride], a, start, stop, step
    )
    return _strings(results, shape, _chars(a), scalar=True)


def mod(a, values):
    a = _vec_string_array(a)
    results, shape = _apply(lambda text: text % values, a)
    widest = max((len(text) for text in results), default=0)
    return _strings(results, shape, widest)


def translate(a, table, deletechars=None):
    a = _vec_string_array(a)
    results, shape = _apply(lambda text: text.translate(table), a)
    return _strings(results, shape, _chars(a))
