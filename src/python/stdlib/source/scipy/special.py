"""shellsim's ``scipy.special``: a small subset of SciPy's special functions.

Star-imports the native ufuncs from ``_scipy_special`` (Rust in
``src/python/stdlib/scipy/special/*.rs``): ``gamma``, ``gammaln``, ``loggamma``, ``psi``
(``digamma``), ``erf``/``erfc`` and their inverses, ``ndtr``/``log_ndtr``/``ndtri``, the
regularized incomplete gamma and beta functions and their inverses, and the two-argument Hurwitz
``zeta``. Everything else below is a composition of those kernels (or of plain NumPy ufuncs)
written directly in this frozen module, following an existing-ufuncs-only style: no elementwise
Python loops.

Kept: ``gamma``, ``gammaln``, ``loggamma``, ``digamma``/``psi``, ``beta``, ``betaln``, ``comb``,
``binom``, ``factorial``, ``erf``, ``erfc``, ``erfinv``, ``erfcinv``, ``ndtr``, ``ndtri``,
``log_ndtr``, ``expit``, ``logit``, ``gammainc``, ``gammaincc``, ``gammaincinv``,
``gammainccinv``, ``betainc``, ``betaincc``, ``betaincinv``, ``zeta``, ``xlogy``, ``xlog1py``,
``entr``, ``rel_entr``, ``kl_div``, ``logsumexp``, ``softmax``, ``expm1``, ``log1p``, ``boxcox``,
``inv_boxcox``. Dropped relative to SciPy: ``rgamma``, ``poch``, ``perm``, ``factorial2``,
``factorialk``, ``log_expit``, ``log_softmax``, and the distribution functions (``bdtr``,
``chdtr``, ``fdtr``, ``pdtr``, ``stdtr`` and their inverses) -- ``scipy.stats`` computes those
directly from ``gammainc``/``betainc`` instead of going through named `scipy.special` wrappers.
Also dropped: the binomial-trial ``DeprecationWarning`` and exact ``reduce``/``accumulate``
frontier text SciPy's own ufuncs carry, which do not apply once these are plain functions.
"""

import numpy as np

from _scipy_special import *  # noqa: F401,F403
from _scipy_special import zeta as _zeta

# `psi` and `digamma` are the same ufunc object under two names, so `special.psi is
# special.digamma` holds.
digamma = psi  # noqa: F821


def _out(value):
    value = np.asarray(value)
    return value[()] if value.ndim == 0 else value


def zeta(x, q=None):
    """Riemann zeta function (`q=None`) or Hurwitz zeta function (`q` given, `x >= 1`)."""
    if q is None:
        return _out(_zeta(np.asarray(x, dtype=np.float64), 1.0))
    x = np.asarray(x, dtype=np.float64)
    result = _zeta(x, np.asarray(q, dtype=np.float64))
    return _out(np.where(x < 1.0, np.nan, result))


def expm1(x):
    """`exp(x) - 1`, accurate near `x = 0`. `scipy.special.expm1` is a separate object from
    NumPy's own `expm1` ufunc of the same name; both compute the same thing, but NumPy's warns
    on overflow at `x = -inf`/large `x` where SciPy's (like the rest of this module) stays
    silent, so this wraps it with the matching local `errstate` suppression."""
    with np.errstate(over="ignore"):
        return np.expm1(x)


def log1p(x):
    """`ln(1 + x)`, accurate near `x = 0`, silent at its domain edges (`log1p(-1) = -inf`,
    `log1p(x < -1) = nan`) the same way `expm1` is."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.log1p(x)


def expit(x):
    """`1 / (1 + exp(-x))`, computed via `exp(-|x|)` so it never overflows either tail."""
    x = np.asarray(x, dtype=np.float64)
    e = np.exp(-np.abs(x))
    return _out(np.where(x >= 0.0, 1.0 / (1.0 + e), e / (1.0 + e)))


def logit(x):
    """`ln(x / (1 - x))`, `nan` outside `[0, 1]`."""
    x = np.asarray(x, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        result = np.log(x) - np.log1p(-x)
        result = np.where((x < 0.0) | (x > 1.0), np.nan, result)
    return _out(result)


def xlogy(x, y):
    """`x * ln(y)`, with `xlogy(0, y) = 0` even where `ln(y)` is `-inf`, except
    `xlogy(0, nan) = nan`."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        result = np.where(x == 0.0, np.where(np.isnan(y), np.nan, 0.0), x * np.log(y))
    return _out(result)


def xlog1py(x, y):
    """`x * ln(1 + y)`, with the same `x == 0` convention as `xlogy`."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        result = np.where(x == 0.0, np.where(np.isnan(y), np.nan, 0.0), x * np.log1p(y))
    return _out(result)


def entr(x):
    """`-x * ln(x)` for `x > 0`, continued to `entr(0) = 0` and `entr(x) = -inf` for `x < 0`."""
    x = np.asarray(x, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        result = np.where(x > 0.0, -(x * np.log(x)), np.where(x == 0.0, 0.0, -np.inf))
    return _out(np.where(np.isnan(x), np.nan, result))


def _log_ratio(x, y):
    """`ln(x / y)` for `x, y > 0`, via `log1p((x - y) / y)` so `x` close to `y` (the common case
    in a Kullback-Leibler-style sum over nearly matching distributions) does not lose precision
    to the cancellation plain `log(x / y)` has there."""
    return np.log1p((x - y) / y)


def rel_entr(x, y):
    """`x * ln(x / y)` for `x, y > 0`, continued to `rel_entr(0, y) = 0` for `y >= 0` and `inf`
    everywhere else."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        generic = x * _log_ratio(x, y)
        result = np.where((x > 0.0) & (y > 0.0), generic, np.where((x == 0.0) & (y >= 0.0), 0.0, np.inf))
    return _out(np.where(np.isnan(x) | np.isnan(y), np.nan, result))


def kl_div(x, y):
    """`x * ln(x / y) - x + y` for `x, y > 0`, continued to `y` for `x == 0, y >= 0` and `inf`
    everywhere else."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        generic = x * _log_ratio(x, y) - x + y
        result = np.where((x > 0.0) & (y > 0.0), generic, np.where((x == 0.0) & (y >= 0.0), y, np.inf))
    return _out(np.where(np.isnan(x) | np.isnan(y), np.nan, result))


def boxcox(x, lmbda):
    """`(x**lmbda - 1) / lmbda`, continued to `ln(x)` at `lmbda == 0`."""
    x = np.asarray(x, dtype=np.float64)
    lmbda = np.asarray(lmbda, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        result = np.where(lmbda == 0.0, np.log(x), np.expm1(lmbda * np.log(x)) / lmbda)
    return _out(result)


def inv_boxcox(y, lmbda):
    """The inverse of `boxcox`: `exp(y)` at `lmbda == 0`, else `(lmbda * y + 1) ** (1 / lmbda)`."""
    y = np.asarray(y, dtype=np.float64)
    lmbda = np.asarray(lmbda, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        result = np.where(lmbda == 0.0, np.exp(y), (lmbda * y + 1.0) ** (1.0 / lmbda))
    return _out(result)


def betaln(a, b):
    """`ln |B(a, b)| = gammaln(a) + gammaln(b) - gammaln(a + b)`."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return _out(gammaln(a) + gammaln(b) - gammaln(a + b))  # noqa: F821


def beta(a, b):
    """`B(a, b) = Gamma(a) Gamma(b) / Gamma(a + b)`, via `betaln`'s magnitude and the actual
    (signed) `gamma` values for the sign, so large arguments (`beta(1e5, 3)`) do not overflow an
    intermediate `Gamma` value the final ratio would bring back into range."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    with np.errstate(over="ignore", invalid="ignore"):
        magnitude = np.exp(betaln(a, b))
        sign = np.sign(gamma(a)) * np.sign(gamma(b)) * np.sign(gamma(a + b))  # noqa: F821
    return _out(magnitude * sign)


def _is_pole(x):
    return (x <= 0.0) & (x == np.floor(x))


def _binom_counting(n, k):
    """`C(n, k)` for integral `0 <= k <= n` by the running product `r = r * (n - k + i) / i`.

    Each step's `r` is itself a binomial coefficient, so the product stays an exact integer while
    it fits a float64 mantissa, and `comb(5, 2)` is exactly `10.0`. Past `k = 1030` every result
    has overflowed to `inf`, which bounds the loop.
    """
    k = np.minimum(k, n - k)
    result = np.ones(np.shape(k))
    steps = min(int(np.max(k)), 1030) if np.size(k) else 0
    for i in range(1, steps + 1):
        result = np.where(k >= i, result * (n - k + i) / i, result)
    return result


def binom(n, k):
    """The generalized binomial coefficient `C(n, k) = Gamma(n + 1) / (Gamma(k + 1)
    Gamma(n - k + 1))`, defined for any real `n` and `k` (not just non-negative integers)."""
    n = np.asarray(n, dtype=np.float64)
    k = np.asarray(k, dtype=np.float64)
    p, q1, q2 = n + 1.0, k + 1.0, n - k + 1.0
    p_pole, q_pole = _is_pole(p), _is_pole(q1) | _is_pole(q2)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        magnitude = np.exp(gammaln(p) - gammaln(q1) - gammaln(q2))  # noqa: F821
        sign = np.sign(gamma(p)) * np.sign(gamma(q1)) * np.sign(gamma(q2))  # noqa: F821
        result = np.where(q_pole, np.where(p_pole, np.nan, 0.0), magnitude * sign)
        result = np.where(p_pole & ~q_pole, np.inf * sign, result)
        counting = (
            np.isfinite(n) & (n == np.floor(n)) & (k == np.floor(k)) & (k >= 0.0) & (k <= n)
        )
        exact = _binom_counting(np.where(counting, n, 0.0), np.where(counting, k, 0.0))
        result = np.where(counting, exact, result)
    return _out(np.where(k == 0.0, 1.0, result))


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
    result = binom(N, k)
    if np.ndim(N) == 0:
        return np.float64(0.0) if float(np.asarray(N)) < 0 else result
    return np.where(np.asarray(N) < 0, 0.0, result)


def _factorial_int(n):
    if n < 0:
        return 0
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result


def _exact_map(arr, fn):
    if arr.ndim == 0:
        return fn(int(arr))
    return _to_int_array([fn(int(value)) for value in arr.ravel()], arr.shape)


def factorial(n, exact=False, extend="zero"):
    """`n!`, continued to `factorial(x) = Gamma(x + 1)` for non-integer `x` (`extend='complex'`)
    or to `0` for negative `x` (`extend='zero'`, the default)."""
    if extend not in ("zero", "complex"):
        raise ValueError("argument `extend` must be one of ('zero', 'complex')")
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
    arr = np.asarray(n, dtype=np.float64)
    with np.errstate(invalid="ignore"):
        result = np.where(arr < 0.0, 0.0, gamma(np.maximum(arr, 0.0) + 1.0))  # noqa: F821
    return _out(np.where(np.isnan(arr), np.nan, result))


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
