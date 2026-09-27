"""Comparison and dtype helpers that NumPy itself writes in Python.

``isclose``, ``allclose``, ``array_equal``, ``array_equiv`` and ``astype`` follow
``numpy/_core/numeric.py`` so that promotion, broadcasting, NaN handling and 0-d results match
NumPy exactly. ``isdtype`` follows ``numpy/_core/numerictypes.py``.
"""

from _numpy import (
    asanyarray,
    asarray,
    bitwise_and,
    bitwise_or,
    bool_,
    complex64,
    complex128,
    complexfloating,
    dtype,
    float16,
    float32,
    float64,
    floating,
    generic,
    inexact,
    int8,
    int16,
    int32,
    int64,
    integer,
    isfinite,
    isnan,
    isscalar,
    less_equal,
    ndarray,
    number,
    object_,
    result_type,
    signedinteger,
    str_,
    uint8,
    uint16,
    uint32,
    uint64,
    unsignedinteger,
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


def astype(x, dtype, /, *, copy=True, device=None):
    if not (isinstance(x, ndarray) or isscalar(x)):
        raise TypeError(f"Input should be a NumPy array or scalar. It is a {type(x)} instead.")
    if device is not None and device != "cpu":
        raise ValueError(f'Device not understood. Only "cpu" is allowed, but received: {device}')
    return x.astype(dtype, copy=copy)


_SIGNED = (int8, int16, int32, int64)
_UNSIGNED = (uint8, uint16, uint32, uint64)
_FLOATING = (float16, float32, float64)
_COMPLEX = (complex64, complex128)
# The scalar types shellsim implements. Abstract types such as ``np.floating`` are accepted, as
# in NumPy, but belong to no kind.
_SCALAR_TYPES = (
    bool_,
    *_SIGNED,
    *_UNSIGNED,
    *_FLOATING,
    *_COMPLEX,
    str_,
    object_,
    generic,
    number,
    integer,
    signedinteger,
    unsignedinteger,
    inexact,
    floating,
    complexfloating,
)
_KINDS = {
    "bool": (bool_,),
    "signed integer": _SIGNED,
    "unsigned integer": _UNSIGNED,
    "integral": _SIGNED + _UNSIGNED,
    "real floating": _FLOATING,
    "complex floating": _COMPLEX,
    "numeric": _SIGNED + _UNSIGNED + _FLOATING + _COMPLEX,
}


def _scalar_type(value):
    if isinstance(value, dtype):
        return value.type
    for scalar_type in _SCALAR_TYPES:
        if value is scalar_type:
            return value
    return None


def isdtype(dtype, kind):
    scalar_type = _scalar_type(dtype)
    if scalar_type is None:
        raise TypeError(f"dtype argument must be a NumPy dtype, but it is a {type(dtype)}.")
    input_kinds = kind if isinstance(kind, tuple) else (kind,)
    processed_kinds = []
    for kind in input_kinds:
        if isinstance(kind, str):
            if kind not in _KINDS:
                raise ValueError(
                    f"kind argument is a string, but {kind!r} is not a known kind name."
                )
            processed_kinds.extend(_KINDS[kind])
            continue
        kind_type = _scalar_type(kind)
        if kind_type is None:
            raise TypeError(
                "kind argument must be comprised of NumPy dtypes or strings only, but is a "
                f"{type(kind)}."
            )
        processed_kinds.append(kind_type)
    return any(scalar_type is kind for kind in processed_kinds)
