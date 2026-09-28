"""Summary statistics: ``describe``, moments and ranking.

Each function is a plain wrapper around a "core" computation on a single 1-D slice; ``_reduce``
applies that core along an ``axis`` and implements ``nan_policy`` (``'propagate'`` just runs the
core on the data as given, so a NaN propagates only if the core's own arithmetic produces one;
``'omit'`` drops NaNs from each slice first; ``'raise'`` rejects NaN input up front) and
``keepdims``. This is a thin function SciPy's statistics call directly, rather than SciPy's
decorator-based ``_axis_nan_policy`` dispatch, so a warning's stack level is one frame shallower
than SciPy's (see docs/scipy.md).
"""

import math
import warnings
from collections import namedtuple

import numpy as np

__all__ = [
    "describe",
    "moment",
    "skew",
    "kurtosis",
    "mode",
    "sem",
    "zscore",
    "zmap",
    "trim_mean",
    "rankdata",
    "ConstantInputWarning",
    "DegenerateDataWarning",
]


class DegenerateDataWarning(RuntimeWarning):
    """Warns that the input data has a degenerate configuration (SciPy's own warning class)."""


class ConstantInputWarning(DegenerateDataWarning):
    """Warns that the input is constant, so a statistic that needs variation is undefined."""


class SmallSampleWarning(RuntimeWarning):
    """Warns that a sample is too small for the statistic to be defined."""


SMALL_SAMPLE = (
    "One or more sample arguments is too small; all returned values will be NaN. "
    "See documentation for sample size requirements."
)
PRECISION_LOSS = (
    "Precision loss occurred in moment calculation due to catastrophic cancellation. "
    "This occurs when the data are nearly identical. Results may be unreliable."
)


def _tuple_bunch(name, fields, extra_fields=(), methods=None):
    """A SciPy-style result class: a named tuple of ``fields`` that also carries the keyword-only
    ``extra_fields`` as attributes, which its ``repr`` and ``_asdict`` include.

    The class derives from a ``namedtuple`` named ``name + "Base"``, as SciPy's ``TtestResult``
    derives from ``TtestResultBase``, so instances keep an attribute dictionary. ``methods``
    adds further class attributes.
    """

    base = namedtuple(name + "Base", fields)

    def __new__(cls, *args, **kwargs):
        extras = {}
        for field in extra_fields:
            if field not in kwargs:
                raise TypeError(f"missing keyword argument {field!r}")
            extras[field] = kwargs.pop(field)
        result = base.__new__(cls, *args, **kwargs)
        for field, value in extras.items():
            setattr(result, field, value)
        return result

    def _asdict(self):
        items = base._asdict(self)
        for field in extra_fields:
            items[field] = getattr(self, field)
        return items

    def __repr__(self):
        items = ", ".join(f"{field}={value!r}" for field, value in self._asdict().items())
        return f"{type(self).__name__}({items})"

    namespace = {
        "__new__": __new__,
        "_asdict": _asdict,
        "__repr__": __repr__,
        "_extra_fields": tuple(extra_fields),
    }
    namespace.update(methods or {})
    return type(name, (base,), namespace)


DescribeResult = namedtuple(
    "DescribeResult", ["nobs", "minmax", "mean", "variance", "skewness", "kurtosis"]
)
ModeResult = namedtuple("ModeResult", ["mode", "count"])


_NAN_POLICIES = ("propagate", "raise", "omit")


def _check_nan_policy(nan_policy):
    if nan_policy not in _NAN_POLICIES:
        raise ValueError(f"nan_policy must be one of {set(_NAN_POLICIES)}")


def _reduce(core, a, axis, nan_policy, keepdims):
    """Apply ``core(row) -> value`` (or a fixed-size tuple) along ``axis`` of ``a``."""
    _check_nan_policy(nan_policy)
    a = np.asarray(a, dtype=float)
    if axis is None:
        a = a.ravel()
        axis = 0
    if nan_policy == "raise" and np.isnan(a).any():
        raise ValueError("The input contains nan values")

    def row_func(row):
        if nan_policy == "omit":
            row = row[~np.isnan(row)]
        return core(row)

    result = np.apply_along_axis(row_func, axis, a)
    if keepdims:
        result = np.expand_dims(result, axis)
    return result


def _scalarize(value):
    value = np.asarray(value)
    return value[()] if value.ndim == 0 else value


def _checked_moment(row, order, mean):
    """A central moment, with SciPy's catastrophic-cancellation check.

    When the second central moment is exactly zero (the sample is constant, so a moment ratio
    would divide by a quantity that rounded to zero), SciPy warns and reports NaN instead. `skew`
    and `kurtosis` each call this twice (once for the second moment, once for the third or
    fourth), so a constant sample warns twice, matching SciPy.
    """
    deviations = row - mean
    second = np.mean(deviations**2)
    if second == 0.0:
        warnings.warn(PRECISION_LOSS, RuntimeWarning, stacklevel=5)
        return np.float64(np.nan)
    return second if order == 2 else np.mean(deviations**order)


def _skew_1d(row, bias):
    n = row.size
    if n < 1:
        return np.nan
    mean = np.mean(row)
    m2 = _checked_moment(row, 2, mean)
    m3 = _checked_moment(row, 3, mean)
    with np.errstate(invalid="ignore", divide="ignore"):
        g1 = m3 / m2**1.5
    if not bias and n > 2:
        g1 = math.sqrt(n * (n - 1)) / (n - 2) * g1
    elif not bias:
        g1 = np.nan
    return g1


def skew(a, axis=0, bias=True, nan_policy="propagate", keepdims=False):
    return _reduce(lambda row: _skew_1d(row, bias), a, axis, nan_policy, keepdims)


def _kurtosis_1d(row, fisher, bias):
    n = row.size
    if n < 1:
        return np.nan
    mean = np.mean(row)
    m2 = _checked_moment(row, 2, mean)
    m4 = _checked_moment(row, 4, mean)
    with np.errstate(invalid="ignore", divide="ignore"):
        g2 = m4 / m2**2 - 3.0
    if not bias and n > 3:
        g2 = ((n + 1) * (g2 + 3.0) - 3 * (n - 1)) * (n - 1) / ((n - 2) * (n - 3))
    elif not bias:
        g2 = np.nan
    return g2 if fisher else g2 + 3.0


def kurtosis(a, axis=0, fisher=True, bias=True, nan_policy="propagate", keepdims=False):
    return _reduce(lambda row: _kurtosis_1d(row, fisher, bias), a, axis, nan_policy, keepdims)


def _moment_1d(row, order, center):
    if order == 0:
        return 1.0
    c = np.mean(row) if center is None else float(center)
    if order == 1 and center is None:
        return 0.0
    return np.mean((row - c) ** order)


def moment(a, order=1, axis=0, nan_policy="propagate", *, center=None, keepdims=False):
    scalar_order = np.ndim(order) == 0
    orders = [int(n) for n in np.atleast_1d(order)]
    outputs = [
        _reduce(lambda row, n=n: _moment_1d(row, n, center), a, axis, nan_policy, keepdims)
        for n in orders
    ]
    return outputs[0] if scalar_order else np.array(outputs)


def describe(a, axis=0, ddof=1, bias=True, nan_policy="propagate"):
    _check_nan_policy(nan_policy)
    a = np.asarray(a, dtype=float)
    if axis is None:
        a = a.ravel()
        axis = 0
    if a.shape[axis] == 0:
        raise ValueError("The input must not be empty.")
    if nan_policy == "raise" and np.isnan(a).any():
        raise ValueError("The input contains nan values")
    if nan_policy == "omit" and np.isnan(a).any():
        raise NotImplementedError(
            "describe(..., nan_policy='omit') with NaN input is not supported by shellsim's SciPy"
        )
    nobs = np.int64(a.shape[axis])
    minmax = (_scalarize(np.min(a, axis=axis)), _scalarize(np.max(a, axis=axis)))
    mean = _scalarize(np.mean(a, axis=axis))
    variance = _scalarize(np.var(a, axis=axis, ddof=ddof))
    skewness = _scalarize(skew(a, axis=axis, bias=bias))
    kurt = _scalarize(kurtosis(a, axis=axis, bias=bias))
    return DescribeResult(nobs, minmax, mean, variance, skewness, kurt)


def sem(a, axis=0, ddof=1, nan_policy="propagate"):
    def core(row):
        n = row.size
        if n - ddof <= 0:
            warnings.warn(SMALL_SAMPLE, SmallSampleWarning, stacklevel=4)
            return np.nan
        return np.std(row, ddof=ddof) / math.sqrt(n)

    return _scalarize(_reduce(core, a, axis, nan_policy, keepdims=False))


def zscore(a, axis=0, ddof=0, nan_policy="propagate"):
    _check_nan_policy(nan_policy)
    a = np.asarray(a, dtype=float)
    if nan_policy == "raise" and np.isnan(a).any():
        raise ValueError("The input contains nan values")

    def core(row):
        if nan_policy == "omit":
            valid = ~np.isnan(row)
            mean = np.mean(row[valid]) if valid.any() else np.nan
            std = np.std(row[valid], ddof=ddof) if valid.any() else np.nan
        else:
            mean = np.mean(row)
            std = np.std(row, ddof=ddof)
        return (row - mean) / std

    return np.apply_along_axis(core, axis, a)


def zmap(scores, compare, axis=0, ddof=0, nan_policy="propagate"):
    scores = np.asarray(scores, dtype=float)
    compare = np.asarray(compare, dtype=float)
    mean = np.mean(compare, axis=axis, keepdims=True)
    std = np.std(compare, axis=axis, ddof=ddof, keepdims=True)
    return (scores - mean) / std


def trim_mean(a, proportiontocut, axis=0):
    def core(row):
        row = np.sort(row)
        n = row.size
        cut = int(np.floor(n * proportiontocut))
        trimmed = row[cut : n - cut] if n - 2 * cut > 0 else row[0:0]
        return np.mean(trimmed)

    return _scalarize(_reduce(core, a, axis, "propagate", keepdims=False))


def _rankdata_1d(row, method):
    n = row.size
    if n == 0:
        return row.astype(float)
    if method == "ordinal":
        order = np.argsort(row, kind="stable")
        ranks = np.empty(n, dtype=float)
        ranks[order] = np.arange(1, n + 1, dtype=float)
        return ranks
    values, inverse, counts = np.unique(row, return_inverse=True, return_counts=True)
    ends = np.cumsum(counts).astype(float)
    starts = ends - counts + 1.0
    if method == "min":
        per_value = starts
    elif method == "max":
        per_value = ends
    elif method == "dense":
        per_value = np.arange(1, len(values) + 1, dtype=float)
    elif method == "average":
        per_value = (starts + ends) / 2.0
    else:
        raise ValueError(f'unknown method "{method}"')
    return per_value[inverse]


def rankdata(a, method="average", axis=None, nan_policy="propagate"):
    if method not in ("average", "min", "max", "dense", "ordinal"):
        raise ValueError(f'unknown method "{method}"')
    _check_nan_policy(nan_policy)
    a = np.asarray(a, dtype=float)
    flat = axis is None
    if flat:
        a = a.ravel()
        axis = 0
    if nan_policy == "raise" and np.isnan(a).any():
        raise ValueError("The input contains nan values")

    def core(row):
        if nan_policy == "omit":
            valid = ~np.isnan(row)
            result = np.full(row.shape, np.nan)
            result[valid] = _rankdata_1d(row[valid], method)
            return result
        return _rankdata_1d(row, method)

    return np.apply_along_axis(core, axis, a)


def mode(a, axis=0, nan_policy="propagate", keepdims=False):
    _check_nan_policy(nan_policy)
    a = np.asarray(a)
    flat = axis is None
    if flat:
        a = a.ravel()
        axis = 0
    if nan_policy == "raise" and a.dtype.kind == "f" and np.isnan(a).any():
        raise ValueError("The input contains nan values")

    def core(row):
        if nan_policy == "omit" and row.dtype.kind == "f":
            row = row[~np.isnan(row)]
        if row.size == 0:
            warnings.warn(SMALL_SAMPLE, SmallSampleWarning, stacklevel=4)
            return np.array([np.nan, 0.0])
        values, counts = np.unique(row, return_counts=True)
        i = np.argmax(counts)
        return np.array([values[i], counts[i]])

    result = np.apply_along_axis(core, axis, a)
    mode_vals = np.take(result, 0, axis=axis)
    counts = np.take(result, 1, axis=axis).astype(np.int64)
    if keepdims:
        mode_vals = np.expand_dims(mode_vals, axis)
        counts = np.expand_dims(counts, axis)
    return ModeResult(_scalarize(mode_vals), _scalarize(counts))
