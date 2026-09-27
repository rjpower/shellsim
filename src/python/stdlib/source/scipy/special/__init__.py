"""scipy.special: shellsim's special-function surface.

Star-imports the native ufuncs from `_ufuncs` (itself a star-import of the native module
`_scipy_special`; see `src/python/stdlib/scipy/special/mod.rs`) and adds the handful of names
SciPy itself defines in Python rather than as a ufunc: the `digamma` alias, the two-argument
`zeta` dispatcher, and the counting/combinatorics and log-sum-exp family (`comb`, `perm`,
`factorial`, `factorial2`, `factorialk`, `logsumexp`, `softmax`, `log_softmax`).
"""

import warnings

import numpy as np

from . import _ufuncs
from ._ufuncs import *  # noqa: F401,F403

# `psi` and `digamma` are the same ufunc object under two names, so `special.psi is
# special.digamma` holds.
digamma = psi  # noqa: F821


def zeta(x, q=None, out=None):
    """Riemann zeta function (`q=None`) or Hurwitz zeta function (`q` given)."""
    if q is None:
        return _ufuncs._riemann_zeta(x, out=out)
    return _ufuncs._zeta(x, q, out=out)


def _warn_non_integer_n():
    """Issue the `DeprecationWarning` a non-integer `n` (trial count) triggers in `bdtr`/`bdtrc`.

    Called from native code (`numpy::ufunc::evaluate`) via `errstate::call_warning`, mirroring
    the warning SciPy's own binomial ufuncs raise for a non-integer or single-precision trial
    count.
    """
    warnings.warn(
        "non-integer arg n is deprecated, removed in SciPy 1.7.x",
        DeprecationWarning,
        stacklevel=2,
    )


def _is_integer_dtype(value):
    return np.issubdtype(np.asarray(value).dtype, np.integer)


def _is_integer_valued(arr):
    return bool(np.all(arr == np.floor(arr)) and np.all(np.isfinite(arr)))


def _to_int_array(values, shape):
    """Build the smallest of an `int64` or an object array of Python `int` that holds `values`."""
    if all(-(2**63) <= value < 2**63 for value in values):
        array = np.empty(len(values), dtype=np.int64)
    else:
        array = np.empty(len(values), dtype=object)
    for index, value in enumerate(values):
        array[index] = value
    return array.reshape(shape)


def _exact_map(arr, fn):
    """Apply an exact (arbitrary-precision) integer `fn` elementwise, preserving scalar-ness."""
    if arr.ndim == 0:
        return fn(int(arr))
    return _to_int_array([fn(int(value)) for value in arr.ravel()], arr.shape)


def _float_map(arr, fn):
    """Apply a `float` -> `float` `fn` elementwise, preserving scalar-ness as `np.float64`."""
    if arr.ndim == 0:
        return np.float64(fn(float(arr)))
    values = [fn(float(value)) for value in arr.ravel()]
    return np.array(values, dtype=np.float64).reshape(arr.shape)


def _comb_int(n, k):
    if k < 0 or n < 0 or k > n:
        return 0
    k = min(k, n - k)
    result = 1
    for i in range(k):
        result = result * (n - i) // (i + 1)
    return result


def comb(N, k, exact=False, repetition=False):
    """Number of combinations of `N` things taken `k` at a time.

    `exact=True` returns an arbitrary-precision Python `int` (or an object array of them);
    otherwise this is `binom(N, k)`, restricted to `0` for negative `N` (unlike the fully
    generalized `binom`, `comb` counts choices from an actual collection of `N` items).
    """
    if repetition:
        return comb(N + k - 1, k, exact=exact)
    if exact:
        n_arr = np.asarray(N)
        k_arr = np.asarray(k)
        if not (_is_integer_valued(n_arr) and _is_integer_valued(k_arr)):
            raise ValueError("Non-integer `N` and `k` with `exact=True` is not supported.")
        n_b, k_b = np.broadcast_arrays(n_arr, k_arr)
        if n_b.ndim == 0:
            return _comb_int(int(n_b), int(k_b))
        values = [_comb_int(int(n), int(k)) for n, k in zip(n_b.ravel(), k_b.ravel())]
        return _to_int_array(values, n_b.shape)
    result = binom(N, k)  # noqa: F821
    if np.ndim(N) == 0:
        return np.float64(0.0) if float(np.asarray(N)) < 0 else result
    return np.where(np.asarray(N) < 0, 0.0, result)


def _perm_int(n, k):
    if k < 0 or n < 0 or k > n:
        return 0
    result = 1
    for i in range(k):
        result *= n - i
    return result


def perm(N, k, exact=False):
    """Number of permutations of `N` things taken `k` at a time."""
    if exact:
        n_arr = np.asarray(N)
        k_arr = np.asarray(k)
        if not (_is_integer_valued(n_arr) and _is_integer_valued(k_arr)):
            raise ValueError("Non-integer `N` and `k` with `exact=True` is not supported.")
        n_b, k_b = np.broadcast_arrays(n_arr, k_arr)
        if n_b.ndim == 0:
            return _perm_int(int(n_b), int(k_b))
        values = [_perm_int(int(n), int(k)) for n, k in zip(n_b.ravel(), k_b.ravel())]
        return _to_int_array(values, n_b.shape)
    # poch(N - k + 1, k) = Gamma(N + 1) / Gamma(N - k + 1), the falling factorial.
    return poch(np.asarray(N) - np.asarray(k) + 1.0, k)  # noqa: F821


_EXTEND_MESSAGE = "argument `extend` must be one of ('zero', 'complex')"


def _check_extend(extend):
    if extend not in ("zero", "complex"):
        raise ValueError(_EXTEND_MESSAGE)


def _factorial_int(n):
    if n < 0:
        return 0
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result


def factorial(n, exact=False, extend="zero"):
    """`n!`, continued to `factorial(x) = Gamma(x + 1)` for non-integer `x` (`extend='complex'`)
    or to `0` for negative `x` (`extend='zero'`, the default)."""
    _check_extend(extend)
    if exact:
        arr = np.asarray(n)
        if not _is_integer_valued(arr):
            raise ValueError("`exact=True` only supports integers")
        return _exact_map(arr, _factorial_int)
    if extend == "complex":
        # No `dtype=` here: complex input must reach the `gamma` ufunc itself so it raises
        # shellsim's own "complex input ... is not supported" error, rather than `np.asarray`
        # rejecting it first with a generic `TypeError`.
        return gamma(np.asarray(n) + 1.0)  # noqa: F821
    arr = np.asarray(n)
    return _float_map(arr, lambda x: 0.0 if x < 0.0 else gamma(x + 1.0))  # noqa: F821


def _multifactorial_int(n, k):
    if n < 0:
        return 0
    result = 1
    while n > 0:
        result *= n
        n -= k
    return result


def _multifactorial_continuous_scalar(z, k):
    """`z!^(k)` continued to real `z` via `k^((z-1)/k) Gamma(z/k + 1) / Gamma(1/k + 1)`, a
    Gamma-function analytic continuation of the multifactorial recurrence `z!^(k) = z (z-k)!^(k)`.
    `k < 0` needs a complex power of the (negative) base; `k > 0` stays real.
    """
    if z == 0.0:
        return np.complex128(1.0) if k < 0 else np.float64(1.0)
    exponent = (z - 1.0) / k
    if k < 0:
        magnitude = abs(k) ** exponent
        angle = np.pi * exponent
        power = np.complex128(complex(magnitude * np.cos(angle), magnitude * np.sin(angle)))
    else:
        power = k**exponent
    return power * gamma(z / k + 1.0) / gamma(1.0 / k + 1.0)  # noqa: F821


def _multifactorial_continuous(z, k):
    arr = np.asarray(z, dtype=np.float64)
    if arr.ndim == 0:
        return _multifactorial_continuous_scalar(float(arr), k)
    values = [_multifactorial_continuous_scalar(float(v), k) for v in arr.ravel()]
    return np.array(values).reshape(arr.shape)


_FACTORIAL2_MESSAGE = (
    "In order to use non-integer arguments, you must opt into this by passing "
    "`extend='complex'`. Note that this changes the result for all negative arguments "
    "(which by default return 0). Additionally, it will rescale the values of the double "
    "factorial at even integers by a factor of sqrt(2/pi)."
)


def factorial2(n, exact=False, extend="zero"):
    """Double factorial `n!! = n (n-2) (n-4) ... `."""
    _check_extend(extend)
    if extend == "complex":
        return _multifactorial_continuous(n, 2.0)
    if not _is_integer_dtype(n):
        raise ValueError(_FACTORIAL2_MESSAGE)
    arr = np.asarray(n)
    if exact:
        return _exact_map(arr, lambda v: _multifactorial_int(v, 2))
    return _float_map(arr, lambda v: 0.0 if v < 0.0 else float(_multifactorial_int(int(v), 2)))


_FACTORIALK_MESSAGE = (
    "In order to use non-integer arguments, you must opt into this by passing "
    "`extend='complex'`. Note that this changes the result for all negative arguments "
    "(which by default return 0). Additionally, it will perturb the values of the "
    "multifactorial at most positive integers `n`."
)


def factorialk(n, k, exact=False, extend="zero"):
    """`k`-fold factorial `n(!^k) = n (n-k) (n-2k) ... `."""
    _check_extend(extend)
    if extend == "complex":
        if k == 0:
            raise ValueError("Parameter k cannot be zero!")
        return _multifactorial_continuous(n, float(k))
    if not (isinstance(k, (int, np.integer)) and not isinstance(k, bool)):
        raise ValueError(_FACTORIALK_MESSAGE)
    if k <= 0:
        raise ValueError(f"For `extend='zero'`, k must be a positive integer, received: {k}")
    if not _is_integer_dtype(n):
        raise ValueError(_FACTORIALK_MESSAGE)
    arr = np.asarray(n)
    if exact:
        return _exact_map(arr, lambda v: _multifactorial_int(v, k))
    return _float_map(arr, lambda v: 0.0 if v < 0.0 else float(_multifactorial_int(int(v), k)))


def logsumexp(a, axis=None, b=None, keepdims=False, return_sign=False):
    """`log(sum(b * exp(a)))`, computed so a large `a` does not overflow `exp`."""
    a = np.asarray(a, dtype=np.float64)
    a_max = np.max(a, axis=axis, keepdims=True)
    a_max_safe = np.where(np.isfinite(a_max), a_max, 0.0)
    if b is not None:
        terms = np.asarray(b, dtype=np.float64) * np.exp(a - a_max_safe)
    else:
        terms = np.exp(a - a_max_safe)
    total = np.sum(terms, axis=axis, keepdims=keepdims)
    reduced_max = a_max if keepdims else np.squeeze(a_max, axis=axis)
    # `total` is legitimately `0.0` when every `a` is `-inf` (or, with `b`, by cancellation), and
    # `log(0.0) = -inf` is exactly the wanted result there, not a domain error to warn about.
    with np.errstate(divide="ignore"):
        if return_sign:
            sign = np.sign(total)
            result = np.log(np.abs(total)) + reduced_max
            return result, sign
        return np.log(total) + reduced_max


def softmax(x, axis=None):
    """`exp(x) / sum(exp(x))`, computed so a large `x` does not overflow `exp`."""
    x = np.asarray(x, dtype=np.float64)
    shifted = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return shifted / np.sum(shifted, axis=axis, keepdims=True)


def log_softmax(x, axis=None):
    """`log(softmax(x))`, computed directly so it stays accurate where `softmax` underflows."""
    x = np.asarray(x, dtype=np.float64)
    shifted = x - np.max(x, axis=axis, keepdims=True)
    return shifted - np.log(np.sum(np.exp(shifted), axis=axis, keepdims=True))
