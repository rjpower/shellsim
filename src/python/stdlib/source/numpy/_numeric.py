"""Comparison helpers that NumPy itself writes in Python on top of ufuncs.

``isclose``, ``allclose``, ``array_equal`` and ``array_equiv`` follow ``numpy/_core/numeric.py``
so that promotion, broadcasting, NaN handling and 0-d results match NumPy exactly.
"""

from _numpy import (
    asanyarray,
    asarray,
    bitwise_and,
    bitwise_or,
    isfinite,
    isnan,
    less_equal,
    ndarray,
    result_type,
)
from _numpy_shape import ravel
from numpy._errstate import errstate


def _all_true(values):
    return all(ravel(values).tolist())


def isclose(a, b, rtol=1.0e-5, atol=1.0e-8, equal_nan=False):
    x, y, atol, rtol = (
        value if isinstance(value, (int, float, complex)) else asanyarray(value)
        for value in (a, b, atol, rtol)
    )
    if getattr(y, "dtype", None) is not None:
        y = asanyarray(y, dtype=result_type(y, 1.0))
    elif isinstance(y, int):
        y = float(y)
    with errstate(invalid="ignore"):
        close = less_equal(abs(x - y), atol + rtol * abs(y))
        result = bitwise_or(bitwise_and(close, isfinite(y)), x == y)
        if equal_nan:
            result = bitwise_or(result, bitwise_and(isnan(x), isnan(y)))
    # A 0-d result becomes a scalar; scalar operands already give one.
    return result[()] if isinstance(result, ndarray) else result


def allclose(a, b, rtol=1.0e-5, atol=1.0e-8, equal_nan=False):
    return _all_true(isclose(a, b, rtol=rtol, atol=atol, equal_nan=equal_nan))


def array_equal(a1, a2, equal_nan=False):
    try:
        a1, a2 = asarray(a1), asarray(a2)
    except Exception:
        return False
    if a1.shape != a2.shape:
        return False
    if not equal_nan:
        return _all_true(asanyarray(a1 == a2))
    if a1 is a2:
        return True
    a1_nan, a2_nan = isnan(a1), isnan(a2)
    if not _all_true(a1_nan == a2_nan):
        return False
    return _all_true(a1[~a1_nan] == a2[~a1_nan])


def array_equiv(a1, a2):
    try:
        a1, a2 = asarray(a1), asarray(a2)
    except Exception:
        return False
    try:
        return _all_true(asanyarray(a1 == a2))
    except ValueError:
        return False
