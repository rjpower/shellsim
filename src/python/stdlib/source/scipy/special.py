"""shellsim's ``scipy.special``.

The special functions are ``numpy.ufunc`` values from the native ``_scipy_special`` module.
This module adds the functions SciPy writes in Python: ``comb``, ``perm``, ``factorial``,
``zeta``, ``softmax``, ``log_softmax`` and ``logsumexp``, following SciPy 1.18's code for NumPy
inputs.
"""

import math
import warnings

import numpy as np
from _scipy_special import *
from _scipy_special import _riemann_zeta, _zeta


def _warn_non_integer_n():
    """Warn as SciPy's legacy ``bdtr`` and ``bdtrc`` loops do for a floating-point ``n``."""
    warnings.warn(
        "non-integer arg n is deprecated, removed in SciPy 1.7.x", DeprecationWarning, stacklevel=2
    )


def zeta(x, q=None, out=None):
    """The Riemann zeta function, or the Hurwitz zeta function when ``q`` is given."""
    if q is None:
        return _riemann_zeta(x, out)
    return _zeta(x, q, out)


def _comb_int(N, k):
    N = int(N)
    k = int(k)
    if k > N or N < 0 or k < 0:
        return 0
    M = N + 1
    numerator = 1
    denominator = 1
    for j in range(1, min(k, N - k) + 1):
        numerator *= M - j
        denominator *= j
    return numerator // denominator


def comb(N, k, *, exact=False, repetition=False):
    """The number of combinations of ``N`` things taken ``k`` at a time."""
    if repetition:
        # C(n, 0) with repetition is 1 for n >= 0; comb(n - 1, 0) would give 0 for n = 0.
        if exact:
            if k == 0 and int(N) == N and N >= 0:
                return 1
        else:
            k, N = np.asarray(k), np.asarray(N)
            cond = (k == 0) & (N >= 0)
            vals = binom(N + k - 1, k)
            if isinstance(vals, np.ndarray):
                vals[cond] = 1.0
            elif cond:
                vals = np.float64(1.0)
            return vals
        return comb(N + k - 1, k, exact=exact)
    if exact:
        if int(N) == N and int(k) == k:
            return _comb_int(N, k)
        raise ValueError("Non-integer `N` and `k` with `exact=True` is not supported.")
    k, N = np.asarray(k), np.asarray(N)
    cond = (k <= N) & (N >= 0) & (k >= 0)
    vals = binom(N, k)
    if isinstance(vals, np.ndarray):
        vals[~cond] = 0
    elif not cond:
        vals = np.float64(0)
    return vals


def perm(N, k, exact=False):
    """The number of permutations of ``N`` things taken ``k`` at a time."""
    if exact:
        N = np.squeeze(N)[()]
        k = np.squeeze(k)[()]
        if not (np.isscalar(N) and np.isscalar(k)):
            raise ValueError("`N` and `k` must be scalar integers with `exact=True`.")
        floor_N, floor_k = int(N), int(k)
        if not (floor_N == N and floor_k == k):
            raise ValueError("Non-integer `N` and `k` with `exact=True` is not supported.")
        if (k > N) or (N < 0) or (k < 0):
            return 0
        val = 1
        for i in range(floor_N - floor_k + 1, floor_N + 1):
            val *= i
        return val
    k, N = np.asarray(k), np.asarray(N)
    cond = (k <= N) & (N >= 0) & (k >= 0)
    vals = poch(N - k + 1, k)
    if isinstance(vals, np.ndarray):
        vals[~cond] = 0
    elif not cond:
        vals = np.float64(0)
    return vals


def _gamma1p(vals):
    """``gamma(n + 1)``, with NaN rather than infinity at -1."""
    res = gamma(vals + 1)
    if isinstance(res, np.ndarray):
        res[vals == -1] = np.nan
    elif np.isinf(res) and vals == -1:
        res = np.float64("nan")
    return res


# The largest n whose factorial fits in int64 and int32.
_FACTORIAL_LIMIT_64BITS = 20
_FACTORIAL_LIMIT_32BITS = 12


def _factorial_array_exact(n):
    un = np.unique(n)
    if un[-1] > _FACTORIAL_LIMIT_64BITS:
        dt = object
    else:
        # SciPy picks int64 above the int32 limit and C long below it; both are int64 on
        # the Linux systems shellsim simulates.
        dt = np.int64
    out = np.empty_like(n, dtype=dt)
    un = un[un > 1]
    out[n < 2] = 1
    out[n < 0] = 0
    if un.size:
        val = math.factorial(int(un[0]))
        out[n == un[0]] = val
        for i in range(len(un) - 1):
            prev = un[i]
            current = un[i + 1]
            for factor in range(int(prev) + 1, int(current) + 1):
                val *= factor
            out[n == current] = val
    return out


def factorial(n, exact=False, extend="zero"):
    """The factorial of ``n``, or ``gamma(n + 1)`` for non-integer ``n``."""
    if extend not in ("zero", "complex"):
        raise ValueError(
            f"argument `extend` must be either 'zero' or 'complex', received: {extend}"
        )
    if exact and extend == "complex":
        raise ValueError("Incompatible options: `exact=True` and `extend='complex'`")
    if extend == "complex":
        raise NotImplementedError(
            "factorial with extend='complex' is not supported by shellsim's SciPy"
        )
    unsupported = (
        "Unsupported data type for `n` in factorial: {dtype}\n"
        "Permitted data types are integers and floating point numbers, as well as complex "
        "numbers if `extend='complex' is passed."
    )
    needs_complex = (
        "In order to use non-integer arguments, you must opt into this by passing "
        "`extend='complex'`. Note that this changes the result for all negative arguments "
        "(which by default return 0)."
    )
    exact_not_possible = "`exact=True` only supports integers, cannot use data type {dtype}"

    if np.ndim(n) == 0 and not isinstance(n, np.ndarray):
        if n is not None and not _is_subdtype(type(n), ("i", "f", "c")):
            raise ValueError(unsupported.format(dtype=type(n)))
        if n is not None and _is_subdtype(type(n), ("c",)):
            raise ValueError(needs_complex)
        if n is None or np.isnan(n):
            return np.float64("nan")
        if n < 0:
            return 0 if exact else np.float64(0)
        if n in {0, 1}:
            return 1 if exact else np.float64(1)
        if exact and _is_subdtype(type(n), ("i",)):
            return math.factorial(int(n))
        if exact:
            raise ValueError(exact_not_possible.format(dtype=type(n)))
        return _gamma1p(n)

    n = np.asarray(n)
    if not _is_subdtype(n.dtype, ("i", "f", "c")):
        raise ValueError(unsupported.format(dtype=n.dtype))
    if _is_subdtype(n.dtype, ("c",)):
        raise ValueError(needs_complex)
    if exact and _is_subdtype(n.dtype, ("f",)):
        raise ValueError(exact_not_possible.format(dtype=n.dtype))
    if n.size == 0:
        return n
    if exact:
        return _factorial_array_exact(n)
    result = np.zeros(n.shape)
    result[np.isnan(n)] = np.nan
    cond = n >= 0
    result[cond] = _gamma1p(n[cond])
    return result


_DTYPE_CLASSES = {"i": np.integer, "f": np.floating, "c": np.complexfloating}


def _is_subdtype(dtype, codes):
    return any(np.issubdtype(dtype, _DTYPE_CLASSES[code]) for code in codes)


def _promote_floating(*args):
    """SciPy's ``xp_promote(..., broadcast=True, force_floating=True)`` for NumPy arrays."""
    args = [np.asarray(arg) if np.iterable(arg) else arg for arg in args]
    present = [arg for arg in args if arg is not None]
    try:
        dtype = np.result_type(*present, 1.0)
    except ValueError:
        dtype = np.result_type(*present, np.asarray(1.0))
    args = [None if arg is None else np.asarray(arg, dtype=dtype) for arg in args]
    present = [arg for arg in args if arg is not None]
    shapes = {arg.shape for arg in present}
    try:
        shape = np.broadcast_shapes(*shapes) if len(shapes) != 1 else present[0].shape
    except ValueError as e:
        raise ValueError("Array shapes are incompatible for broadcasting.") from e
    return [
        arg if arg is None or arg.shape == shape else np.broadcast_to(arg, shape)
        for arg in args
    ]


def _wrap_radians(x):
    # Wrap radians to (-pi, pi], preserving relative precision inside that interval.
    wrapped = -((-x + np.pi) % (2 * np.pi) - np.pi)
    return np.where(np.abs(x) < np.pi, x, wrapped)


def _elements_and_indices_with_max_real(a, axis):
    if np.iscomplexobj(a):
        real_a = np.real(a)
        max_ = np.max(real_a, axis=axis, keepdims=True)
        mask = real_a == max_
        # Of the elements with the largest real part, keep the last.
        i = np.reshape(np.arange(a.size), a.shape)
        i = np.where(mask, i, -1)
        max_i = np.max(i, axis=axis, keepdims=True)
        mask = i == max_i
        a = np.where(mask, a, 0.0)
        max_ = np.sum(a, axis=axis, dtype=a.dtype, keepdims=True)
    else:
        max_ = np.max(a, axis=axis, keepdims=True)
        mask = a == max_
    return max_, mask


def _logsumexp(a, b, axis, return_sign):
    # An element adds nothing to the sum when its weight is zero, even if it is infinite.
    if b is not None:
        a = np.where(b == 0, -np.inf, a)
    a_max, i_max = _elements_and_indices_with_max_real(a, axis)
    # The largest terms are summed separately, for precision.
    a = np.where(i_max, -np.inf, a)
    i_max_dt = i_max.astype(a.dtype)
    b_i_max = i_max_dt if b is None else b * i_max_dt
    m = np.sum(b_i_max, axis=axis, keepdims=True, dtype=a.dtype)
    exp = b * np.exp(a - a_max) if b is not None else np.exp(a - a_max)
    s = np.sum(exp, axis=axis, keepdims=True, dtype=exp.dtype)
    s = np.where(s == 0, s, s / m)
    sgn = np.sign(s + 1) * np.sign(m)
    if np.iscomplexobj(s):
        # a_max can carry a phase for complex input.
        sgn = sgn * np.exp(np.imag(a_max) * 1.0j)
    else:
        s = np.where(s < -1, -s - 2, s)
        m = np.abs(m)
    out = np.log1p(s) + np.log(m) + a_max
    if return_sign:
        out = np.real(out)
    elif not np.iscomplexobj(out):
        out = np.where(sgn < 0, np.nan, out)
    return out, sgn


def logsumexp(a, axis=None, b=None, keepdims=False, return_sign=False):
    """``log(sum(b * exp(a)))``, computed without overflow."""
    a, b = _promote_floating(a, b)
    a = np.atleast_1d(a)
    b = np.atleast_1d(b) if b is not None else b
    axis = tuple(range(a.ndim)) if axis is None else axis

    if a.size != 0:
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            # Where the result is infinite, the direct calculation handles the edge cases.
            b_exp_a = np.exp(a) if b is None else b * np.exp(a)
            sum_ = np.sum(b_exp_a, axis=axis, keepdims=True)
            sgn_inf = np.sign(sum_) if return_sign else None
            sum_ = np.abs(sum_) if return_sign else sum_
            out_inf = np.log(sum_)
        with np.errstate(divide="ignore", invalid="ignore"):
            out, sgn = _logsumexp(a, b, axis, return_sign)
        out_finite = np.isfinite(out)
        out = np.where(out_finite, out, out_inf)
        sgn = np.where(out_finite, sgn, sgn_inf) if return_sign else sgn
    else:
        shape = np.asarray(a.shape)
        shape[axis] = 1
        out = np.full(tuple(shape), -np.inf, dtype=a.dtype)
        sgn = np.sign(out)

    if np.iscomplexobj(out):
        if return_sign:
            sgn = np.real(sgn) + _wrap_radians(np.imag(sgn)).astype(sgn.dtype) * 1j
        else:
            out = np.real(out) + _wrap_radians(np.imag(out)).astype(out.dtype) * 1j

    out = np.squeeze(out, axis=axis) if not keepdims else out
    sgn = np.squeeze(sgn, axis=axis) if (sgn is not None and not keepdims) else sgn
    out = out[()] if out.ndim == 0 else out
    sgn = sgn[()] if (sgn is not None and sgn.ndim == 0) else sgn
    return (out, sgn) if return_sign else out


def softmax(x, axis=None):
    """``exp(x) / sum(exp(x))`` along ``axis``."""
    x = np.asarray(x)
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x_shifted = np.exp(x - x_max)
    return exp_x_shifted / np.sum(exp_x_shifted, axis=axis, keepdims=True)


def log_softmax(x, axis=None):
    """``log(softmax(x))``, computed without overflow."""
    x = np.asarray(x)
    x_max = np.max(x, axis=axis, keepdims=True)
    if x_max.ndim > 0:
        x_max = np.where(np.isfinite(x_max), x_max, 0)
    elif not np.isfinite(x_max):
        x_max = 0
    tmp = x - x_max
    exp_tmp = np.exp(tmp)
    with np.errstate(divide="ignore"):
        s = np.sum(exp_tmp, axis=axis, keepdims=True)
        out = np.log(s)
    return tmp - out
