"""Array methods and statistics NumPy writes on top of ufuncs.

``mean``, ``var``, ``std``, ``ptp`` and ``clip`` follow ``numpy/_core/_methods.py``; ``round``
follows ``PyArray_Round`` in ``numpy/_core/src/multiarray/calculation.c``, which is itself a
sequence of ufunc calls; ``average`` and ``count_nonzero`` follow
``numpy/lib/_function_base_impl.py`` and ``numpy/_core/numeric.py``.
Keeping NumPy's structure keeps its result dtypes, warnings, and float16 handling: integer and
boolean inputs compute in ``float64``, ``float16`` accumulates in ``float32``, and an empty slice
warns and gives NaN.
"""

import operator
import warnings

from _numpy import (
    _AxisError,
    add,
    asanyarray,
    bool_,
    complexfloating,
    copyto,
    empty,
    float16,
    float64,
    floating,
    integer,
    intp,
    issubdtype,
    maximum,
    minimum,
    multiply,
    ndarray,
    positive,
    result_type,
    rint,
    sqrt,
    subtract,
    true_divide,
)
from _numpy import dtype as _dtype
from _numpy_reduce import any as _any
from _numpy_reduce import sum as _sum


def _normalize_axis_index(axis, ndim):
    if not -ndim <= axis < ndim:
        raise _AxisError(f"axis {axis} is out of bounds for array of dimension {ndim}")
    return axis + ndim if axis < 0 else axis


def _count_reduce_items(arr, axis):
    if axis is None:
        axis = tuple(range(arr.ndim))
    elif not isinstance(axis, tuple):
        axis = (axis,)
    items = 1
    for ax in axis:
        items *= arr.shape[_normalize_axis_index(ax, arr.ndim)]
    return intp(items)


def _is_integer_or_bool(dtype):
    return issubdtype(dtype, integer) or issubdtype(dtype, bool_)


def _reject_where(where):
    if where is not True:
        raise NotImplementedError("where= masks are not supported by shellsim's NumPy")


def _mean(a, axis=None, dtype=None, out=None, keepdims=False, *, where=True):
    _reject_where(where)
    arr = asanyarray(a)
    is_float16_result = False
    rcount = _count_reduce_items(arr, axis)
    if rcount == 0:
        warnings.warn("Mean of empty slice", RuntimeWarning, stacklevel=2)
    if dtype is None:
        if _is_integer_or_bool(arr.dtype):
            dtype = _dtype("f8")
        elif issubdtype(arr.dtype, float16):
            dtype = _dtype("f4")
            is_float16_result = True
    ret = add.reduce(arr, axis=axis, dtype=dtype, out=out, keepdims=keepdims)
    if isinstance(ret, ndarray):
        ret = true_divide(ret, rcount, out=ret, casting="unsafe")
        if is_float16_result and out is None:
            ret = ret.astype(arr.dtype)
    elif hasattr(ret, "dtype"):
        if is_float16_result:
            ret = arr.dtype.type(ret / rcount)
        else:
            ret = ret.dtype.type(ret / rcount)
    else:
        ret = ret / rcount
    return ret


def _var(a, axis=None, dtype=None, out=None, ddof=0, keepdims=False, *, where=True, mean=None,
         correction=None):
    _reject_where(where)
    if correction is not None:
        if ddof != 0:
            raise ValueError("ddof and correction can't be provided simultaneously.")
        ddof = correction
    arr = asanyarray(a)
    rcount = _count_reduce_items(arr, axis)
    if ddof >= rcount:
        warnings.warn("Degrees of freedom <= 0 for slice", RuntimeWarning, stacklevel=2)
    if dtype is None and _is_integer_or_bool(arr.dtype):
        dtype = _dtype("f8")
    if mean is not None:
        arrmean = mean
    else:
        arrmean = add.reduce(arr, axis=axis, dtype=dtype, keepdims=True)
        if isinstance(arrmean, ndarray):
            arrmean = true_divide(arrmean, rcount, out=arrmean, casting="unsafe")
        elif hasattr(arrmean, "dtype"):
            arrmean = arrmean.dtype.type(arrmean / rcount)
        else:
            arrmean = arrmean / rcount
    x = asanyarray(arr - arrmean)
    if issubdtype(arr.dtype, floating) or issubdtype(arr.dtype, integer):
        x = multiply(x, x, out=x)
    elif issubdtype(arr.dtype, complexfloating):
        # NumPy's fast path squares the real and imaginary parts separately.
        x = add(multiply(x.real, x.real), multiply(x.imag, x.imag))
    else:
        x = multiply(x, x.conjugate()).real
    ret = add.reduce(x, axis=axis, dtype=dtype, out=out, keepdims=keepdims)
    rcount = maximum(rcount - ddof, 0)
    if isinstance(ret, ndarray):
        ret = true_divide(ret, rcount, out=ret, casting="unsafe")
    elif hasattr(ret, "dtype"):
        ret = ret.dtype.type(ret / rcount)
    else:
        ret = ret / rcount
    return ret


def _std(a, axis=None, dtype=None, out=None, ddof=0, keepdims=False, *, where=True, mean=None,
         correction=None):
    ret = _var(a, axis=axis, dtype=dtype, out=out, ddof=ddof, keepdims=keepdims, where=where,
               mean=mean, correction=correction)
    if isinstance(ret, ndarray):
        ret = sqrt(ret, out=ret)
    elif hasattr(ret, "dtype"):
        ret = ret.dtype.type(sqrt(ret))
    else:
        ret = sqrt(ret)
    return ret


def _ptp(a, axis=None, out=None, keepdims=False):
    return subtract(
        maximum.reduce(a, axis=axis, out=out, keepdims=keepdims),
        minimum.reduce(a, axis=axis, keepdims=keepdims),
        out,
    )


def mean(a, axis=None, dtype=None, out=None, keepdims=False, *, where=True):
    return _mean(a, axis=axis, dtype=dtype, out=out, keepdims=keepdims, where=where)


def var(a, axis=None, dtype=None, out=None, ddof=0, keepdims=False, *, where=True, mean=None,
        correction=None):
    return _var(a, axis=axis, dtype=dtype, out=out, ddof=ddof, keepdims=keepdims, where=where,
                mean=mean, correction=correction)


def std(a, axis=None, dtype=None, out=None, ddof=0, keepdims=False, *, where=True, mean=None,
        correction=None):
    return _std(a, axis=axis, dtype=dtype, out=out, ddof=ddof, keepdims=keepdims, where=where,
                mean=mean, correction=correction)


def ptp(a, axis=None, out=None, keepdims=False):
    return _ptp(asanyarray(a), axis=axis, out=out, keepdims=keepdims)


def average(a, axis=None, weights=None, returned=False, *, keepdims=False):
    a = asanyarray(a)
    if weights is None:
        avg = _mean(a, axis=axis, keepdims=keepdims)
        avg_as_array = asanyarray(avg)
        scl = avg_as_array.dtype.type(a.size / avg_as_array.size)
    else:
        wgt = asanyarray(weights)
        if issubdtype(a.dtype, integer) or issubdtype(a.dtype, bool_):
            result_dtype = result_type(a.dtype, wgt.dtype, "f8")
        else:
            result_dtype = result_type(a.dtype, wgt.dtype)
        if a.shape != wgt.shape:
            if axis is None:
                raise TypeError(
                    "Axis must be specified when shapes of a and weights differ.")
            if wgt.ndim != 1:
                raise TypeError(
                    "1D weights expected when shapes of a and weights differ.")
            axis_index = _normalize_axis_index(axis, a.ndim)
            if wgt.shape[0] != a.shape[axis_index]:
                raise ValueError(
                    "Length of weights not compatible with specified axis.")
            shape = [1] * a.ndim
            shape[axis_index] = wgt.shape[0]
            wgt = wgt.reshape(tuple(shape))
        scl = _sum(wgt, axis=axis, dtype=result_dtype, keepdims=keepdims)
        if _any(scl == 0.0):
            raise ZeroDivisionError("Weights sum to zero, can't be normalized")
        avg = avg_as_array = true_divide(
            _sum(multiply(a, wgt, dtype=result_dtype), axis=axis, keepdims=keepdims), scl)
    if returned:
        if asanyarray(scl).shape != avg_as_array.shape:
            scl = scl + avg_as_array * 0
        return avg, scl
    return avg


def count_nonzero(a, axis=None, *, keepdims=False):
    a = asanyarray(a)
    if a.dtype.kind == "U":
        nonzero = a != ""
    else:
        nonzero = a.astype(bool)
    if axis is None and not keepdims:
        return int(_sum(nonzero))
    return _sum(nonzero, axis=axis, dtype=intp, keepdims=keepdims)


def _power_of_ten(n):
    """``power_of_ten`` in ``calculation.c``: exact up to 1e8, then repeated scaling by ten."""
    if n < 9:
        return (1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8)[n]
    ret = 1e9
    while n > 9:
        ret *= 10.0
        n -= 1
    return ret


def _round_array(a, decimals, out):
    """``PyArray_Round``: scale, round half to even with ``rint``, and scale back."""
    if out is not None and out.size != a.size:
        raise ValueError("invalid output shape")
    if a.dtype.kind == "c":
        arr = out if out is not None else a.copy()
        arr.real[...] = _round_array(asanyarray(a.real), decimals, None)
        arr.imag[...] = _round_array(asanyarray(a.imag), decimals, None)
        return arr
    is_integer = a.dtype.kind in "iu"
    if decimals >= 0:
        if is_integer:
            if out is not None:
                copyto(out, a)
                return out
            return a
        if decimals == 0:
            return rint(a) if out is None else rint(a, out)
        op1, op2 = multiply, true_divide
    else:
        op1, op2 = true_divide, multiply
        decimals = -decimals
    ret_int = False
    if out is None:
        # Integers round through float64 and cast back.
        ret_int = is_integer
        out = empty(a.shape, float64 if is_integer else a.dtype)
    f = _power_of_ten(decimals)
    ret = op1(a, f, out)
    rint(ret, ret)
    op2(ret, f, ret)
    if ret_int:
        return ret.astype(a.dtype)
    return ret


def _round(a, decimals=0, out=None):
    """``ndarray.round``: a 0-d result without ``out`` is returned as a scalar."""
    result = _round_array(a, operator.index(decimals), out)
    if out is None and isinstance(result, ndarray) and result.ndim == 0:
        return result[()]
    return result


def _clip(a, min=None, max=None, out=None, **kwargs):
    if a.dtype.kind in "iu":
        # Python int bounds beyond the dtype's range are dropped rather than converted, as
        # NEP 50 does no value-based promotion.
        bits = a.dtype.itemsize * 8
        low, high = (0, (1 << bits) - 1) if a.dtype.kind == "u" else (
            -(1 << (bits - 1)),
            (1 << (bits - 1)) - 1,
        )
        if type(min) is int and min <= low:
            min = None
        if type(max) is int and max >= high:
            max = None
    if min is None and max is None:
        return positive(a, out=out, **kwargs)
    if min is None:
        return minimum(a, max, out=out, **kwargs)
    if max is None:
        return maximum(a, min, out=out, **kwargs)
    # NumPy's `clip` ufunc resolves all three operands together; `minimum(maximum(...))`
    # promotes the same way and propagates NaN from any operand as its loops do.
    if out is None:
        return minimum(maximum(a, min, **kwargs), max, **kwargs)
    maximum(a, min, out=out, **kwargs)
    return minimum(out, max, out=out, **kwargs)
