"""Summary statistics, correlation and t-tests, following SciPy 1.18's ``scipy/stats/_stats_py.py``.

SciPy writes these functions against the array API; this port is their NumPy branch. Functions
that SciPy wraps with ``_axis_nan_policy_factory`` declare SciPy's full signature here, including
``nan_policy`` and ``keepdims``, and pass their samples to the ``apply`` function the factory
returns (see ``_axis_nan_policy``).

Three paths need features shellsim lacks and fail explicitly: ``nan_policy='omit'`` in
``describe`` and ``spearmanr``, which SciPy hands to ``scipy.stats.mstats`` and masked arrays,
and the resampling ``method`` objects (``PermutationMethod``, ``MonteCarloMethod`` and
``BootstrapMethod``), which are not implemented. Results are ``_make_tuple_bunch`` classes, which
unpack like SciPy's named tuples but are not ``tuple`` instances.
"""

import math
import operator
import warnings

import numpy as np
import scipy.special as special
from scipy._lib._bunch import _make_tuple_bunch
from scipy._lib._util import (
    _contains_nan,
    _count_nonmasked,
    _get_nan,
    _promote,
    _result_type,
    apply_where,
)
from scipy.stats._axis_nan_policy import (
    SmallSampleWarning,
    _axis_nan_policy_factory,
    _broadcast_array_shapes_remove_axis,
    _broadcast_shapes,
    too_small_1d_not_omit,
    too_small_1d_omit,
    too_small_nd_not_omit,
    too_small_nd_omit,
)
from scipy.stats._warnings_errors import ConstantInputWarning, NearConstantInputWarning

__all__ = [
    "chisquare",
    "describe",
    "kurtosis",
    "linregress",
    "mode",
    "moment",
    "pearsonr",
    "power_divergence",
    "rankdata",
    "sem",
    "skew",
    "spearmanr",
    "trim_mean",
    "ttest_1samp",
    "ttest_ind",
    "ttest_ind_from_stats",
    "ttest_rel",
    "zmap",
    "zscore",
]


def _chk_asarray(a, axis):
    if axis is None:
        a = np.reshape(a, (-1,))
        outaxis = 0
    else:
        a = np.asarray(a)
        outaxis = axis

    if a.ndim == 0:
        a = np.reshape(a, (-1,))

    return a, outaxis


def _identity(x):
    return x


def _single_output(x, _):
    return (x,)


SignificanceResult = _make_tuple_bunch("SignificanceResult", ["statistic", "pvalue"], [])


def _pack_CorrelationResult(statistic, pvalue, correlation):
    res = SignificanceResult(statistic, pvalue)
    res.correlation = correlation
    return res


def _unpack_CorrelationResult(res, _):
    return res.statistic, res.pvalue, res.correlation


ModeResult = _make_tuple_bunch("ModeResult", ("mode", "count"))


def _mode_result(mode, count):
    # When a slice is empty, `_axis_nan_policy` automatically produces
    # NaN for `mode` and `count`. This is a reasonable convention for `mode`,
    # but `count` should not be NaN; it should be zero.
    i = np.isnan(count)
    if i.shape == ():
        count = np.asarray(0, dtype=count.dtype)[()] if i else count
    else:
        count = np.where(i, 0, count)
    if mode.ndim > 0:
        # With multidimensional input and NaNs omitted, `_axis_nan_policy` computes each
        # axis-slice separately and returns `count` in the result dtype of `mode` and
        # `count`, so change it back to an integer.
        count = count.astype(np.asarray(0).dtype)
    return ModeResult(mode, count)


_mode_policy = _axis_nan_policy_factory(_mode_result, override={"nan_propagation": False})


def mode(a, axis=0, nan_policy="propagate", keepdims=False):
    """Return an array of the modal (most common) value in the passed array.

    If there is more than one such value, only one is returned. The count of the modal value
    is also returned.
    """
    return _mode_policy(_mode, [a], {}, axis, nan_policy, keepdims)


def _mode(a, axis=0):
    if not np.isdtype(a.dtype, "numeric"):
        message = (
            "Argument `a` is not recognized as numeric. "
            "Support for input that cannot be coerced to a numeric "
            "array was deprecated in SciPy 1.9.0 and removed in SciPy "
            "1.11.0. Please consider `np.unique`."
        )
        raise TypeError(message)

    if a.size == 0:
        NaN = _get_nan(a)
        return ModeResult(*np.asarray([NaN, 0], dtype=NaN.dtype))

    if a.ndim == 1:
        vals, cnts = np.unique_counts(a)
        # in contrast with np.unique, `unique_counts` treats all NaNs as distinct,
        # but we have always grouped them. Replace `cnts` corresponding with NaNs
        # with the number of NaNs.
        mask = np.isnan(vals)
        cnts = np.where(mask, np.count_nonzero(mask), cnts)
        modes, counts = vals[np.argmax(cnts)], np.max(cnts)
        counts = counts.astype(np.asarray(1).dtype)
        modes = modes[()] if modes.ndim == 0 else modes
        counts = counts[()] if counts.ndim == 0 else counts
        return ModeResult(modes, counts)

    # `axis` is always -1 after the `_axis_nan_policy` decorator
    y = np.sort(a, axis=-1)

    # Get boolean array of elements that are different from the previous element
    i = np.concatenate(
        [np.ones(y.shape[:-1] + (1,), dtype=bool), (y[..., :-1] != y[..., 1:]) & ~np.isnan(y[..., :-1])],
        axis=-1,
    )
    # Get linear integer indices of these elements in a raveled array
    indices = np.arange(y.size)[np.reshape(i, (-1,))]
    # The difference between integer indices is the number of repeats
    append = np.full(indices.shape[:-1] + (1,), y.size, dtype=indices.dtype)
    counts = np.diff(indices, append=append)
    # Now we form an array of `counts` corresponding with each element of `y`...
    counts = np.reshape(np.repeat(counts, counts), y.shape)
    # ... so we can get the argmax of *each slice* separately.
    k = np.argmax(counts, axis=-1, keepdims=True)
    # Extract the corresponding element/count, and eliminate the reduced dimension
    modes = np.take_along_axis(y, k, axis=-1)[..., 0]
    counts = np.take_along_axis(counts, k, axis=-1)[..., 0]
    modes = modes[()] if modes.ndim == 0 else modes
    counts = counts[()] if counts.ndim == 0 else counts
    return ModeResult(modes, counts)


def _trim_mean(a, proportiontocut, axis=0):
    a = np.asarray(a)

    if a.size == 0:
        return _get_nan(a)

    if axis is None:
        a = np.reshape(a, (-1,))
        axis = 0

    nobs = _count_nonmasked(a, axis=axis, keepdims=True)
    lowercut = int(proportiontocut * nobs)
    uppercut = nobs - lowercut
    if lowercut > uppercut:
        raise ValueError("Proportion too big.")

    atmp = np.partition(a, (lowercut, uppercut - 1), axis)

    sl = [slice(None)] * atmp.ndim
    sl[axis] = slice(lowercut, uppercut)
    trimmed = atmp[tuple(sl)]

    trimmed = _promote(trimmed, force_floating=True)
    return np.mean(trimmed, axis=axis)


_one_output_policy = _axis_nan_policy_factory(
    _identity, result_to_tuple=_single_output, n_outputs=1
)


def trim_mean(a, proportiontocut, axis=0, *, nan_policy="propagate", keepdims=False):
    """Return mean of array after trimming a specified fraction of extreme values.

    ``proportiontocut`` of the sorted data is removed from each end.
    """
    return _one_output_policy(
        _trim_mean, [a], {"proportiontocut": proportiontocut}, axis, nan_policy, keepdims
    )


#####################################
#              MOMENTS              #
#####################################


def _moment_outputs(kwds, default_order=1):
    order = np.atleast_1d(kwds.get("order", default_order))
    message = "`order` must be a scalar or a non-empty 1D array."
    if order.size == 0 or order.ndim > 1:
        raise ValueError(message)
    return len(order)


def _moment_result_object(*args):
    if len(args) == 1:
        return args[0]
    return np.stack(args)


# When `order` is array-like with size > 1, moment produces an *array*
# rather than a tuple, but the zeroth dimension is to be treated like
# separate outputs. It is important to make the distinction between
# separate outputs when adding the reduced axes back (`keepdims=True`).
def _moment_tuple(x, n_out):
    return tuple(x[i, ...] for i in range(x.shape[0])) if n_out > 1 else (x,)


_moment_policy = _axis_nan_policy_factory(
    _moment_result_object, result_to_tuple=_moment_tuple, n_outputs=_moment_outputs
)


def moment(a, order=1, axis=0, nan_policy="propagate", *, center=None, keepdims=False):
    """Calculate the nth moment about the mean (or ``center``) for a sample."""
    return _moment_policy(
        _moment_function, [a], {"order": order, "center": center}, axis, nan_policy, keepdims
    )


def _moment_function(a, order=1, axis=0, *, center=None):
    a, center, order = _promote(a, center, order, force_floating=True)

    if np.any(order != np.round(order)):
        raise ValueError("All elements of `order` must be integral.")

    # _axis_nan_policy decorator ensures that axis=-1
    if order.ndim > 0:
        order = np.reshape(order, (-1,) + (1,) * a.ndim)
        return _moment(a, order, axis=-1, center=center)
    else:
        res = _moment(a, order, axis=-1, center=center)
        return res[()] if res.ndim == 0 else res


def _demean(a, mean, axis, *, precision_warning=True):
    # subtracts `mean` from `a` and returns the result,
    # warning if there is catastrophic cancellation. `mean`
    # must be the mean of `a` along axis with `keepdims=True`.
    # Used in e.g. `_moment`, `_zscore`, `_xp_var`. See gh-15905.
    a_zero_mean = a - mean

    if a_zero_mean.size == 0 or not precision_warning:
        return a_zero_mean

    eps = np.finfo(mean.dtype).eps * 10

    with np.errstate(divide="ignore", invalid="ignore"):
        rel_diff = np.max(np.abs(a_zero_mean), axis=axis, keepdims=True) / np.abs(mean)

    n = _count_nonmasked(a, axis)
    with np.errstate(invalid="ignore"):
        precision_loss = np.any(np.asarray(rel_diff < eps) & np.asarray(n > 1))

    if precision_loss:
        message = (
            "Precision loss occurred in moment calculation due to "
            "catastrophic cancellation. This occurs when the data "
            "are nearly identical. Results may be unreliable."
        )
        warnings.warn(message, RuntimeWarning, stacklevel=5)
    return a_zero_mean


def _moment(a, order, axis, *, center=None):
    """Vectorized calculation of raw moment about specified center.

    When `center` is None, the center is the mean of the data, and the moment of order 1 is
    exactly zero rather than rounding error.
    """
    order = np.asarray(order, dtype=a.dtype)
    order_0 = order == 0
    order_1 = (order == 1) & (center is None)
    center = np.mean(a, axis=axis, keepdims=True) if center is None else center
    a_zero_mean = _demean(a, center, axis)
    res = np.mean(a_zero_mean**order, axis=axis, keepdims=True)
    if a.shape[-1] > 0 and (np.any(order_0) or np.any(order_1)):
        res = np.where(order_0, np.ones_like(res), res)
        res = np.where(order_1, np.zeros_like(res), res)

    return np.squeeze(res, axis=axis)


def _var(x, axis=0, ddof=0, mean=None):
    # Calculate variance of sample, warning if precision is lost
    var = _moment(x, 2, axis, center=mean)
    if ddof != 0:
        n = _count_nonmasked(x, axis)
        n = np.asarray(n, dtype=x.dtype)
        var *= n / (n - ddof)  # to avoid error on division by zero
    return var


def skew(a, axis=0, bias=True, nan_policy="propagate", *, keepdims=False):
    """Compute the sample skewness of a data set.

    With ``bias=False``, the result is corrected for statistical bias.
    """
    return _one_output_policy(_skew, [a], {"bias": bias}, axis, nan_policy, keepdims)


def _skew(a, axis=0, bias=True):
    a, axis = _chk_asarray(a, axis)
    n = _count_nonmasked(a, axis)

    mean = np.mean(a, axis=axis, keepdims=True)
    mean_reduced = np.squeeze(mean, axis=axis)  # needed later
    m2 = _moment(a, 2, axis, center=mean)
    m3 = _moment(a, 3, axis, center=mean)
    with np.errstate(all="ignore"):
        eps = np.finfo(m2.dtype).eps
        zero = m2 <= (eps * mean_reduced) ** 2
        vals = np.where(zero, np.nan, m3 / m2**1.5)
    if not bias:
        can_correct = ~zero & (n > 2)
        if np.any(can_correct):
            nval = ((n - 1.0) * n) ** 0.5 / (n - 2.0) * m3 / m2**1.5
            vals = np.where(can_correct, nval, vals)

    return vals[()] if vals.ndim == 0 else vals


def kurtosis(a, axis=0, fisher=True, bias=True, nan_policy="propagate", *, keepdims=False):
    """Compute the kurtosis (Fisher or Pearson) of a dataset.

    Fisher's definition subtracts 3.0, so a normal distribution has kurtosis 0.
    """
    return _one_output_policy(
        _kurtosis, [a], {"fisher": fisher, "bias": bias}, axis, nan_policy, keepdims
    )


def _kurtosis(a, axis=0, fisher=True, bias=True):
    a, axis = _chk_asarray(a, axis)

    n = _count_nonmasked(a, axis)
    mean = np.mean(a, axis=axis, keepdims=True)
    mean_reduced = np.squeeze(mean, axis=axis)  # needed later
    m2 = _moment(a, 2, axis, center=mean)
    m4 = _moment(a, 4, axis, center=mean)
    with np.errstate(all="ignore"):
        zero = m2 <= (np.finfo(m2.dtype).eps * mean_reduced) ** 2
        vals = np.where(zero, np.nan, m4 / m2**2.0)

    if not bias:
        can_correct = ~zero & (n > 3)
        if np.any(can_correct):
            nval = 1.0 / (n - 2) / (n - 3) * ((n**2 - 1.0) * m4 / m2**2.0 - 3 * (n - 1) ** 2.0)
            vals = np.where(can_correct, nval + 3.0, vals)

    vals = vals - 3 if fisher else vals
    return vals[()] if vals.ndim == 0 else vals


DescribeResult = _make_tuple_bunch(
    "DescribeResult", ("nobs", "minmax", "mean", "variance", "skewness", "kurtosis")
)


def describe(a, axis=0, ddof=1, bias=True, nan_policy="propagate"):
    """Compute several descriptive statistics of the passed array.

    Returns the number of observations, the minimum and maximum, the mean, the variance (with
    ``ddof`` degrees of freedom), the skewness and the kurtosis.
    """
    a, axis = _chk_asarray(a, axis)

    contains_nan = _contains_nan(a, nan_policy)

    if nan_policy == "omit" and contains_nan:
        raise NotImplementedError(
            "describe(..., nan_policy='omit') with NaN input is not supported by shellsim's "
            "SciPy"
        )

    if a.size == 0:
        raise ValueError("The input must not be empty.")

    n = np.asarray(_count_nonmasked(a, axis), dtype=np.int64)
    n = n[()] if n.ndim == 0 else n
    mm = (np.min(a, axis=axis), np.max(a, axis=axis))
    a = _promote(a, force_floating=True)
    m = np.mean(a, axis=axis)
    v = _var(a, axis=axis, ddof=ddof)
    v = v[()] if v.ndim == 0 else v
    sk = skew(a, axis, bias=bias)
    kurt = kurtosis(a, axis, bias=bias)

    return DescribeResult(n, mm, m, v, sk, kurt)


def _get_pvalue(statistic, distribution, alternative, symmetric=True):
    """Get p-value given the statistic, (continuous) distribution, and alternative."""
    if alternative == "less":
        pvalue = distribution.cdf(statistic)
    elif alternative == "greater":
        pvalue = distribution.sf(statistic)
    elif alternative == "two-sided":
        pvalue = 2 * (
            distribution.sf(np.abs(statistic))
            if symmetric
            else np.minimum(distribution.cdf(statistic), distribution.sf(statistic))
        )
    else:
        message = "`alternative` must be 'less', 'greater', or 'two-sided'."
        raise ValueError(message)

    return pvalue


_sem_policy = _axis_nan_policy_factory(
    _identity, result_to_tuple=_single_output, n_outputs=1, too_small=1
)


def sem(a, axis=0, ddof=1, nan_policy="propagate", *, keepdims=False):
    """Compute standard error of the mean, with ``ddof`` delta degrees of freedom."""
    return _sem_policy(_sem, [a], {"ddof": ddof}, axis, nan_policy, keepdims)


def _sem(a, axis=0, ddof=1):
    if axis is None:
        a = np.reshape(a, (-1,))
        axis = 0
    a = np.atleast_1d(np.asarray(a))
    n = _count_nonmasked(a, axis)
    s = np.std(a, axis=axis, correction=ddof) / n**0.5
    return s


def zscore(a, axis=0, ddof=0, nan_policy="propagate"):
    """Compute the z score of each value relative to the sample mean and standard deviation."""
    return zmap(a, a, axis=axis, ddof=ddof, nan_policy=nan_policy)


def zmap(scores, compare, axis=0, ddof=0, nan_policy="propagate"):
    """Calculate the relative z-scores of ``scores`` against the mean and deviation of
    ``compare``."""
    like_zscore = scores is compare
    scores, compare = _promote(scores, compare, force_floating=True)

    with warnings.catch_warnings():
        if like_zscore:  # zscore should not emit SmallSampleWarning
            warnings.simplefilter("ignore", SmallSampleWarning)

        mn = _xp_mean(compare, axis=axis, keepdims=True, nan_policy=nan_policy)
        std = (
            _xp_var(compare, axis=axis, correction=ddof, keepdims=True, nan_policy=nan_policy)
            ** 0.5
        )

    with np.errstate(invalid="ignore", divide="ignore"):
        z = _demean(scores, mn, axis, precision_warning=False) / std

    # If we know that scores and compare are identical, we can infer that
    # some slices should have NaNs.
    if like_zscore:
        eps = np.finfo(z.dtype).eps
        zero = std <= np.abs(eps * mn)
        zero = np.broadcast_to(zero, z.shape)
        z = np.where(zero, np.nan, z)

    return z


#####################################
#       CORRELATION FUNCTIONS       #
#####################################


class _SimpleChi2:
    # A very simple chi-squared distribution for use in hypothesis tests.
    def __init__(self, df):
        self.df = df

    def cdf(self, x):
        return special.chdtr(self.df, x)

    def sf(self, x):
        return special.chdtrc(self.df, x)


class _SimpleBeta:
    # A very simple beta distribution for use in hypothesis tests.
    def __init__(self, a, b, *, loc=None, scale=None):
        self.a = a
        self.b = b
        self.loc = loc
        self.scale = scale

    def cdf(self, x):
        if self.loc is not None or self.scale is not None:
            loc = 0 if self.loc is None else self.loc
            scale = 1 if self.scale is None else self.scale
            return special.betainc(self.a, self.b, (x - loc) / scale)
        return special.betainc(self.a, self.b, x)

    def sf(self, x):
        if self.loc is not None or self.scale is not None:
            loc = 0 if self.loc is None else self.loc
            scale = 1 if self.scale is None else self.scale
            return special.betaincc(self.a, self.b, (x - loc) / scale)
        return special.betaincc(self.a, self.b, x)


class _SimpleStudentT:
    # A very simple t distribution for use in hypothesis tests.
    def __init__(self, df):
        self.df = df

    def cdf(self, t):
        return special.stdtr(self.df, t)

    def sf(self, t):
        return special.stdtr(self.df, -t)


def _pearsonr_fisher_ci(r, n, confidence_level, alternative):
    """Compute the confidence interval for Pearson's R by the Fisher transformation."""
    r = np.asarray(r)
    ones = np.ones_like(r)
    n = np.asarray(n, dtype=r.dtype)
    confidence_level = np.asarray(confidence_level, dtype=r.dtype)

    with np.errstate(divide="ignore", invalid="ignore"):
        zr = np.arctanh(r)
        se = np.sqrt(1 / (n - 3))

    if alternative == "two-sided":
        h = special.ndtri(0.5 + confidence_level / 2)
        zlo = zr - h * se
        zhi = zr + h * se
        rlo = np.tanh(zlo)
        rhi = np.tanh(zhi)
    elif alternative == "less":
        h = special.ndtri(confidence_level)
        zhi = zr + h * se
        rhi = np.tanh(zhi)
        rlo = -ones
    else:
        # alternative == "greater":
        h = special.ndtri(confidence_level)
        zlo = zr - h * se
        rlo = np.tanh(zlo)
        rhi = ones

    mask = n <= 3
    rlo = np.where(mask, -1.0, rlo)
    rhi = np.where(mask, 1.0, rhi)

    rlo = rlo[()] if rlo.ndim == 0 else rlo
    rhi = rhi[()] if rhi.ndim == 0 else rhi
    return ConfidenceInterval(low=rlo, high=rhi)


ConfidenceInterval = _make_tuple_bunch("ConfidenceInterval", ["low", "high"])

PearsonRResultBase = _make_tuple_bunch("PearsonRResultBase", ["statistic", "pvalue"], [])


class PearsonRResult(PearsonRResultBase):
    """Result of `scipy.stats.pearsonr`: the statistic, p-value and a confidence interval."""

    def __init__(self, statistic, pvalue, alternative, n, x, y, axis):
        super().__init__(statistic, pvalue)
        self._alternative = alternative
        self._n = n
        self._x = x
        self._y = y
        self._axis = axis

        # add alias for consistency with other correlation functions
        self.correlation = statistic

    def confidence_interval(self, confidence_level=0.95, method=None):
        """The confidence interval for the correlation coefficient.

        The interval comes from the Fisher transformation. Bootstrap intervals are not
        implemented.
        """
        if method is not None:
            message = "`method` must be an instance of `BootstrapMethod` or None."
            raise ValueError(message)
        return _pearsonr_fisher_ci(self.statistic, self._n, confidence_level, self._alternative)


def pearsonr(x, y, *, alternative="two-sided", method=None, axis=0):
    """Pearson correlation coefficient and p-value for testing non-correlation.

    The p-value comes from the exact beta distribution of the coefficient under the null
    hypothesis of independent normal samples.
    """
    x, y = _promote(x, y, force_floating=True)
    dtype = x.dtype

    if axis is None:
        x = np.reshape(x, (-1,))
        y = np.reshape(y, (-1,))
        axis = -1

    axis_int = int(axis)
    if axis_int != axis:
        raise ValueError("`axis` must be an integer.")
    axis = axis_int

    try:
        np.broadcast_shapes(x.shape, y.shape)
        # For consistency with other `stats` functions, we need to
        # match the dimensionalities before looking at `axis`.
        ndim = max(x.ndim, y.ndim)
        x = np.reshape(x, (1,) * (ndim - x.ndim) + x.shape)
        y = np.reshape(y, (1,) * (ndim - y.ndim) + y.shape)

    except (ValueError, RuntimeError) as e:
        message = "`x` and `y` must be broadcastable."
        raise ValueError(message) from e

    if x.shape[axis] != y.shape[axis]:
        raise ValueError("`x` and `y` must have the same length along `axis`.")

    if x.shape[axis] < 2:
        raise ValueError("`x` and `y` must have length at least 2.")

    n = np.asarray(_count_nonmasked(x, axis=axis), dtype=x.dtype)

    x = np.moveaxis(x, axis, -1)
    y = np.moveaxis(y, axis, -1)
    axis = -1

    if np.isdtype(dtype, "complex floating"):
        raise ValueError("This function does not support complex data")

    x = x.astype(dtype, copy=False)
    y = y.astype(dtype, copy=False)
    threshold = np.finfo(dtype).eps ** 0.75

    # If an input is constant, the correlation coefficient is not defined.
    const_x = np.all(x == x[..., 0:1], axis=-1)
    const_y = np.all(y == y[..., 0:1], axis=-1)
    const_xy = const_x | const_y

    any_const_xy = np.any(const_xy)
    if any_const_xy:
        msg = "An input array is constant; the correlation coefficient is not defined."
        warnings.warn(ConstantInputWarning(msg), stacklevel=2)
        x = np.where(const_x[..., np.newaxis], np.nan, x)
        y = np.where(const_y[..., np.newaxis], np.nan, y)

    if method is not None:
        message = "`method` must be an instance of `PermutationMethod`, `MonteCarloMethod`, or None."
        raise ValueError(message)

    xmean = np.mean(x, axis=axis, keepdims=True)
    ymean = np.mean(y, axis=axis, keepdims=True)
    xm = x - xmean
    ym = y - ymean

    # Scale by the largest magnitude first to avoid premature overflow in the norms,
    # e.g. of [-5e210, 5e210, 3e200, -3e200].
    xmax = np.max(np.abs(xm), axis=axis, keepdims=True)
    ymax = np.max(np.abs(ym), axis=axis, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        normxm = xmax * np.linalg.norm(xm / xmax, ord=2, axis=axis, keepdims=True)
        normym = ymax * np.linalg.norm(ym / ymax, ord=2, axis=axis, keepdims=True)

    nconst_x = np.any(normxm < threshold * np.abs(xmean), axis=axis)
    nconst_y = np.any(normym < threshold * np.abs(ymean), axis=axis)
    nconst_xy = nconst_x | nconst_y
    if np.any(nconst_xy & (~const_xy)):
        # If all the values in x (likewise y) are very close to the mean,
        # the loss of precision that occurs in the subtraction xm = x - xmean
        # might result in large errors in r.
        msg = (
            "An input array is nearly constant; the computed "
            "correlation coefficient may be inaccurate."
        )
        warnings.warn(NearConstantInputWarning(msg), stacklevel=2)

    with np.errstate(invalid="ignore", divide="ignore"):
        r = np.vecdot(xm / normxm, ym / normym, axis=axis)

    # Presumably, if abs(r) > 1, then it is only some small artifact of
    # floating point arithmetic.
    r = np.clip(r, -1.0, 1.0)
    r = np.where(const_xy, np.nan, r)

    # As explained in the docstring, the distribution of `r` under the null
    # hypothesis is the beta distribution on (-1, 1) with a = b = n/2 - 1.
    ab = np.asarray(n / 2 - 1, dtype=dtype)
    dist = _SimpleBeta(ab, ab, loc=-1, scale=2)
    pvalue = _get_pvalue(r, dist, alternative)

    mask = n == 2  # return exactly 1.0 or -1.0 values for n == 2 case as promised

    def special_case(r):
        return np.where(np.isnan(r), np.nan, np.ones_like(r))

    r = apply_where(mask, r, np.round, fill_value=r)
    pvalue = apply_where(mask, (r,), special_case, fill_value=pvalue)

    r = r[()] if r.ndim == 0 else r
    pvalue = pvalue[()] if pvalue.ndim == 0 else pvalue
    return PearsonRResult(
        statistic=r, pvalue=pvalue, n=n, alternative=alternative, x=x, y=y, axis=axis
    )


def spearmanr(a, b=None, axis=0, nan_policy="propagate", alternative="two-sided"):
    """Calculate a Spearman correlation coefficient with associated p-value.

    The coefficient is the Pearson correlation of the ranks. The p-value uses a t distribution
    with ``n - 2`` degrees of freedom.
    """
    if axis is not None and axis > 1:
        raise ValueError(
            "spearmanr only handles 1-D or 2-D arrays, "
            f"supplied axis argument {axis}, please use only "
            "values 0, 1 or None for axis"
        )

    a, axisout = _chk_asarray(a, axis)
    if a.ndim > 2:
        raise ValueError("spearmanr only handles 1-D or 2-D arrays")

    if b is None:
        if a.ndim < 2:
            raise ValueError("`spearmanr` needs at least 2 variables to compare")
    else:
        # Concatenate a and b, so that we now only have to handle the case
        # of a 2-D `a`.
        b, _ = _chk_asarray(b, axis)
        if axisout == 0:
            a = np.column_stack((a, b))
        else:
            a = np.vstack((a, b))

    n_vars = a.shape[1 - axisout]
    n_obs = a.shape[axisout]
    if n_obs <= 1:
        # Handle empty arrays or single observations.
        res = SignificanceResult(np.nan, np.nan)
        res.correlation = np.nan
        return res

    warn_msg = "An input array is constant; the correlation coefficient is not defined."

    constant_axis = False
    if axisout == 0:
        constant_columns = np.all(a == a[0, :], axis=0)
        constant_axis = np.any(constant_columns)
        if constant_axis:
            # If an input is constant, the correlation coefficient
            # is not defined.
            warnings.warn(ConstantInputWarning(warn_msg), stacklevel=2)
    else:  # case when axisout == 1 b/c a is 2 dim only
        constant_rows = np.all(a.T == a.T[0, :], axis=0)
        constant_axis = np.any(constant_rows)
        if constant_axis:
            # If an input is constant, the correlation coefficient
            # is not defined.
            warnings.warn(ConstantInputWarning(warn_msg), stacklevel=2)

    a_contains_nan = _contains_nan(a, nan_policy)
    variable_has_nan = np.zeros(n_vars, dtype=bool)
    if a_contains_nan:
        if nan_policy == "omit":
            raise NotImplementedError(
                "spearmanr(..., nan_policy='omit') with NaN input is not supported by "
                "shellsim's SciPy"
            )
        elif nan_policy == "propagate":
            if a.ndim == 1 or n_vars <= 2:
                res = SignificanceResult(np.nan, np.nan)
                res.correlation = np.nan
                return res
            else:
                # Keep track of variables with NaNs, set the outputs to NaN
                # only for those variables
                variable_has_nan = np.isnan(a).any(axis=axisout)

    a_ranked = np.apply_along_axis(rankdata, axisout, a)

    if constant_axis:
        with np.errstate(invalid="ignore"):
            rs = np.corrcoef(a_ranked, rowvar=axisout)
    else:
        rs = np.corrcoef(a_ranked, rowvar=axisout)

    dof = n_obs - 2  # degrees of freedom

    # rs can have elements equal to 1, so avoid zero division warnings
    with np.errstate(divide="ignore"):
        # clip the small negative values possibly caused by rounding
        # errors before taking the square root
        t = rs * np.sqrt((dof / ((rs + 1.0) * (1.0 - rs))).clip(0))

    dist = _SimpleStudentT(dof)
    prob = _get_pvalue(t, dist, alternative)

    # For backwards compatibility, return scalars when comparing 2 columns
    if rs.shape == (2, 2):
        res = SignificanceResult(rs[1, 0], prob[1, 0])
        res.correlation = rs[1, 0]
        return res
    else:
        rs[variable_has_nan, :] = np.nan
        rs[:, variable_has_nan] = np.nan
        res = SignificanceResult(rs[()], prob[()])
        res.correlation = rs
        return res


#####################################
#       INFERENTIAL STATISTICS      #
#####################################

TtestResultBase = _make_tuple_bunch("TtestResultBase", ["statistic", "pvalue"], ["df"])


class TtestResult(TtestResultBase):
    """Result of a t-test: the statistic, p-value, degrees of freedom and a confidence interval."""

    def __init__(self, statistic, pvalue, df, alternative, standard_error, estimate):
        super().__init__(statistic, pvalue, df=df)
        self._alternative = alternative
        self._standard_error = standard_error  # denominator of t-statistic
        self._estimate = estimate  # point estimate of sample mean
        self._dtype = np.asarray(statistic).dtype

    def confidence_interval(self, confidence_level=0.95):
        """The confidence interval for the population mean, or difference of means."""
        low, high = _t_confidence_interval(
            self.df, self.statistic, confidence_level, self._alternative, self._dtype
        )
        low = low * self._standard_error + self._estimate
        high = high * self._standard_error + self._estimate
        return ConfidenceInterval(low=low, high=high)


def pack_TtestResult(statistic, pvalue, df, alternative, standard_error, estimate):
    # Due to behavior of `_axis_nan_policy` decorator, `alternative` can be any number
    # of dimensions, but there is at most one unique non-NaN value.
    alternative = np.asarray(alternative)
    alternative = (
        _xp_mean(alternative, axis=None, nan_policy="omit", warn=False)
        if alternative.size != 0
        else np.nan
    )
    return TtestResult(
        statistic,
        pvalue,
        df=df,
        alternative=alternative,
        standard_error=standard_error,
        estimate=estimate,
    )


def unpack_TtestResult(res, _):
    return (
        res.statistic,
        res.pvalue,
        res.df,
        res._alternative,
        res._standard_error,
        res._estimate,
    )


_ttest_policy = _axis_nan_policy_factory(
    pack_TtestResult, result_to_tuple=unpack_TtestResult, n_outputs=6
)
_ttest_paired_policy = _axis_nan_policy_factory(
    pack_TtestResult, result_to_tuple=unpack_TtestResult, n_outputs=6, paired=True
)


def ttest_1samp(
    a, popmean, axis=0, nan_policy="propagate", alternative="two-sided", *, keepdims=False
):
    """Calculate the T-test for the mean of ONE group of scores against ``popmean``."""
    return _ttest_policy(
        _ttest_1samp, [a, popmean], {"alternative": alternative}, axis, nan_policy, keepdims
    )


def _ttest_1samp(a, popmean, axis=0, alternative="two-sided"):
    a, popmean = _promote(a, popmean, force_floating=True)
    a, axis = _chk_asarray(a, axis)

    n = _count_nonmasked(a, axis)
    df = n - 1

    if a.shape[axis] == 0:
        # This is really only needed for *testing* _axis_nan_policy decorator
        # It won't happen when the decorator is used.
        NaN = _get_nan(a)
        return TtestResult(NaN, NaN, df=NaN, alternative=NaN, standard_error=NaN, estimate=NaN)

    mean = np.mean(a, axis=axis)
    try:
        popmean = np.asarray(popmean)
        popmean = np.squeeze(popmean, axis=axis) if popmean.ndim > 0 else popmean
    except ValueError as e:
        raise ValueError("`popmean.shape[axis]` must equal 1.") from e
    d = mean - popmean
    v = _var(a, axis=axis, ddof=1)
    denom = np.sqrt(v / n)

    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.divide(d, denom)
        t = t[()] if t.ndim == 0 else t

    dist = _SimpleStudentT(np.asarray(df, dtype=t.dtype))
    prob = _get_pvalue(t, dist, alternative)
    prob = prob[()] if prob.ndim == 0 else prob

    # when nan_policy='omit', `df` can be different for different axis-slices
    df = np.broadcast_to(np.asarray(df), t.shape)
    df = df[()] if df.ndim == 0 else df
    # _axis_nan_policy decorator doesn't play well with strings
    alternative_num = {"less": -1, "two-sided": 0, "greater": 1}[alternative]
    return TtestResult(
        t, prob, df=df, alternative=alternative_num, standard_error=denom, estimate=mean
    )


def _t_confidence_interval(df, t, confidence_level, alternative, dtype=None):
    # Input validation on `alternative` is already done
    # We just need IV on confidence_level
    dtype = np.asarray(t).dtype if dtype is None else dtype

    if confidence_level < 0 or confidence_level > 1:
        message = "`confidence_level` must be a number between 0 and 1."
        raise ValueError(message)

    confidence_level = np.asarray(confidence_level, dtype=dtype)
    inf = np.asarray(np.inf, dtype=dtype)

    if alternative < 0:  # 'less'
        p = confidence_level
        low, high = np.broadcast_arrays(-inf, special.stdtrit(df, p))
    elif alternative > 0:  # 'greater'
        p = 1 - confidence_level
        low, high = np.broadcast_arrays(special.stdtrit(df, p), inf)
    elif alternative == 0:  # 'two-sided'
        tail_probability = (1 - confidence_level) / 2
        p = np.stack((tail_probability, 1 - tail_probability))
        # axis of p must be the zeroth and orthogonal to all the rest
        p = np.reshape(p, tuple([2] + [1] * np.asarray(df).ndim))
        ci = special.stdtrit(df, p)
        low, high = ci[0, ...], ci[1, ...]
    else:  # alternative is NaN when input is empty (see _axis_nan_policy)
        nan = np.asarray(np.nan)
        p, nans = np.broadcast_arrays(t, nan)
        low, high = nans, nans

    low = np.asarray(low, dtype=dtype)
    low = low[()] if low.ndim == 0 else low
    high = np.asarray(high, dtype=dtype)
    high = high[()] if high.ndim == 0 else high
    return low, high


def _ttest_ind_from_stats(mean1, mean2, denom, df, alternative):
    d = mean1 - mean2
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.divide(d, denom)

    dist = _SimpleStudentT(np.asarray(df, dtype=t.dtype))
    prob = _get_pvalue(t, dist, alternative)
    prob = prob[()] if prob.ndim == 0 else prob

    t = t[()] if t.ndim == 0 else t
    prob = prob[()] if prob.ndim == 0 else prob
    return t, prob


def _unequal_var_ttest_denom(v1, n1, v2, n2):
    vn1 = v1 / n1
    vn2 = v2 / n2
    with np.errstate(divide="ignore", invalid="ignore"):
        df = (vn1 + vn2) ** 2 / (vn1**2 / (n1 - 1) + vn2**2 / (n2 - 1))

    # If df is undefined, variances are zero (assumes n1 > 0 & n2 > 0).
    # Hence it doesn't matter what df is as long as it's not NaN.
    df = np.where(np.isnan(df), 1.0, df)
    denom = np.sqrt(vn1 + vn2)
    return df, denom


def _equal_var_ttest_denom(v1, n1, v2, n2):
    # If there is a single observation in one sample, this formula for pooled
    # variance breaks down because the variance of that sample is undefined.
    # The pooled variance is still defined, though, because the (n-1) in the
    # numerator should cancel with the (n-1) in the denominator, leaving only
    # the sum of squared differences from the mean: zero.
    v1 = np.where(np.asarray(n1 == 1), 0.0, v1)
    v2 = np.where(np.asarray(n2 == 1), 0.0, v2)

    df = n1 + n2 - 2.0
    svar = ((n1 - 1) * v1 + (n2 - 1) * v2) / df
    denom = np.sqrt(svar * (1.0 / n1 + 1.0 / n2))
    df = np.asarray(df, dtype=denom.dtype)
    return df, denom


Ttest_indResult = _make_tuple_bunch("Ttest_indResult", ("statistic", "pvalue"))


def ttest_ind_from_stats(
    mean1, std1, nobs1, mean2, std2, nobs2, equal_var=True, alternative="two-sided"
):
    """T-test for means of two independent samples from descriptive statistics."""
    mean1, std1, nobs1, mean2, std2, nobs2 = _promote(
        mean1, std1, nobs1, mean2, std2, nobs2, force_floating=True
    )

    if equal_var:
        df, denom = _equal_var_ttest_denom(std1**2, nobs1, std2**2, nobs2)
    else:
        df, denom = _unequal_var_ttest_denom(std1**2, nobs1, std2**2, nobs2)

    res = _ttest_ind_from_stats(mean1, mean2, denom, df, alternative)
    return Ttest_indResult(*res)


def ttest_ind(
    a,
    b,
    *,
    axis=0,
    equal_var=True,
    nan_policy="propagate",
    alternative="two-sided",
    trim=0,
    method=None,
    keepdims=False,
):
    """Calculate the T-test for the means of *two independent* samples of scores.

    ``equal_var=False`` performs Welch's t-test, and ``trim`` Yuen's trimmed t-test.
    """
    kwds = {"equal_var": equal_var, "alternative": alternative, "trim": trim, "method": method}
    return _ttest_policy(_ttest_ind, [a, b], kwds, axis, nan_policy, keepdims)


def _ttest_ind(a, b, axis=0, equal_var=True, alternative="two-sided", trim=0, method=None):
    a, b = _promote(a, b, force_floating=True)

    if axis is None:
        a, b, axis = np.reshape(a, (-1,)), np.reshape(b, (-1,)), 0

    if not (0 <= trim < 0.5):
        raise ValueError("Trimming percentage should be 0 <= `trim` < .5.")

    if method is not None:
        message = (
            "`method` must be an instance of `PermutationMethod`, an instance "
            "of `MonteCarloMethod`, or None (default)."
        )
        raise ValueError(message)

    result_shape = _broadcast_array_shapes_remove_axis((a, b), axis=axis)
    NaN = _get_nan(a, b, shape=result_shape)
    if a.size == 0 or b.size == 0:
        return TtestResult(NaN, NaN, df=NaN, alternative=NaN, standard_error=NaN, estimate=NaN)

    alternative_nums = {"less": -1, "two-sided": 0, "greater": 1}

    n1 = _count_nonmasked(a, axis)
    n2 = _count_nonmasked(b, axis)

    if trim == 0:
        with np.errstate(divide="ignore", invalid="ignore"):
            v1 = _var(a, axis, ddof=1)
            v2 = _var(b, axis, ddof=1)

        m1 = np.mean(a, axis=axis)
        m2 = np.mean(b, axis=axis)
    else:
        v1, m1, n1 = _ttest_trim_var_mean_len(a, trim, axis)
        v2, m2, n2 = _ttest_trim_var_mean_len(b, trim, axis)

    if equal_var:
        df, denom = _equal_var_ttest_denom(v1, n1, v2, n2)
    else:
        df, denom = _unequal_var_ttest_denom(v1, n1, v2, n2)

    t, prob = _ttest_ind_from_stats(m1, m2, denom, df, alternative)

    # when nan_policy='omit', `df` can be different for different axis-slices
    df = np.broadcast_to(df, np.shape(t))
    df = df[()] if df.ndim == 0 else df
    estimate = m1 - m2

    return TtestResult(
        t,
        prob,
        df=df,
        alternative=alternative_nums[alternative],
        standard_error=denom,
        estimate=estimate,
    )


def _ttest_trim_var_mean_len(a, trim, axis):
    """Variance, mean and length of ``a`` for Yuen's trimmed t-test."""
    # further calculations in this test assume that the inputs are sorted.
    a = np.sort(a, axis=axis)

    # `g` is the number of elements to be replaced on each tail, converted
    # from a percentage amount of trimming
    n = a.shape[axis]
    g = int(n * trim)

    # Calculate the Winsorized variance of the input samples according to
    # specified `g`
    v = _calculate_winsorized_variance(a, g, axis)

    # the total number of elements in the trimmed samples
    n -= 2 * g

    # calculate the g-times trimmed mean
    m = trim_mean(a, trim, axis=axis)
    return v, m, n


def _calculate_winsorized_variance(a, g, axis):
    """Calculate the Winsorized variance of the sorted input ``a``."""
    if g == 0:
        return _var(a, ddof=1, axis=axis)
    # move the intended axis to the end that way it is easier to manipulate
    a_win = np.moveaxis(a, axis, -1).copy()

    # save where NaNs are for later use.
    nans_indices = np.any(np.isnan(a_win), axis=-1)

    # Replace the g smallest and g largest values with the nearest remaining ones.
    a_win[..., :g] = a_win[..., g : g + 1]
    a_win[..., -g:] = a_win[..., -g - 1 : -g]

    # The degrees of freedom are h - 1 with h = n - 2g, which is ddof = 2g + 1.
    var_win = np.asarray(_var(a_win, ddof=(2 * g + 1), axis=-1))

    # with `nan_policy='propagate'`, NaNs may be completely trimmed out
    # because they were sorted into the tail of the array. In these cases,
    # replace computed variances with `np.nan`.
    var_win = np.where(nans_indices, np.nan, var_win)
    return var_win


def ttest_rel(
    a, b, axis=0, nan_policy="propagate", alternative="two-sided", *, keepdims=False
):
    """Calculate the t-test on TWO RELATED samples of scores, a and b."""
    return _ttest_paired_policy(
        _ttest_rel, [a, b], {"alternative": alternative}, axis, nan_policy, keepdims
    )


def _ttest_rel(a, b, axis=0, alternative="two-sided"):
    return _ttest_1samp(a - b, popmean=0.0, axis=axis, alternative=alternative)


# Map from names to lambda_ values used in power_divergence().
_power_div_lambda_names = {
    "pearson": 1,
    "log-likelihood": 0,
    "freeman-tukey": -0.5,
    "mod-log-likelihood": -1,
    "neyman": -2,
    "cressie-read": 2 / 3,
}


Power_divergenceResult = _make_tuple_bunch("Power_divergenceResult", ("statistic", "pvalue"))

_power_divergence_policy = _axis_nan_policy_factory(
    Power_divergenceResult, paired=True, too_small=-1
)


def power_divergence(
    f_obs, f_exp=None, ddof=0, axis=0, lambda_=None, *, nan_policy="propagate", keepdims=False
):
    """Cressie-Read power divergence statistic and goodness of fit test."""
    samples = [f_obs] if f_exp is None else [f_obs, f_exp]
    kwds = {"ddof": ddof, "lambda_": lambda_}
    return _power_divergence_policy(
        _power_divergence, samples, kwds, axis, nan_policy, keepdims
    )


def chisquare(
    f_obs, f_exp=None, ddof=0, axis=0, *, sum_check=True, nan_policy="propagate", keepdims=False
):
    """Perform Pearson's chi-squared test."""
    samples = [f_obs] if f_exp is None else [f_obs, f_exp]
    kwds = {"ddof": ddof, "lambda_": "pearson", "sum_check": sum_check}
    return _power_divergence_policy(
        _power_divergence, samples, kwds, axis, nan_policy, keepdims
    )


def _power_divergence(f_obs, f_exp=None, ddof=0, axis=0, lambda_=None, sum_check=True):
    f_obs, f_exp, ddof = _promote(f_obs, f_exp, ddof, force_floating=True)

    # Convert the input argument `lambda_` to a numerical value.
    if isinstance(lambda_, str):
        if lambda_ not in _power_div_lambda_names:
            names = repr(list(_power_div_lambda_names.keys()))[1:-1]
            raise ValueError(
                f"invalid string for lambda_: {lambda_!r}. Valid strings are {names}"
            )
        lambda_ = _power_div_lambda_names[lambda_]
    elif lambda_ is None:
        lambda_ = 1

    if f_exp is not None:
        # not sure why we force to float64, but not going to touch it
        f_obs_float = np.asarray(f_obs, dtype=np.float64)
        bshape = _broadcast_shapes((f_obs_float.shape, f_exp.shape))
        f_obs_float = np.broadcast_to(f_obs_float, bshape)
        f_exp = np.broadcast_to(f_exp, bshape)

        if sum_check:
            dtype_res = np.result_type(f_obs.dtype, f_exp.dtype)
            rtol = np.finfo(dtype_res).eps ** 0.5  # to pass existing tests
            with np.errstate(invalid="ignore"):
                f_obs_sum = np.sum(f_obs_float, axis=axis, keepdims=True)
                f_exp_sum = np.sum(f_exp, axis=axis, keepdims=True)
                relative_diff = np.abs(f_obs_sum - f_exp_sum) / np.minimum(f_obs_sum, f_exp_sum)
                diff_gt_tol = np.any(relative_diff > rtol, axis=axis, keepdims=True)

            if np.any(diff_gt_tol):
                msg = (
                    f"For each axis slice, the sum of the observed "
                    f"frequencies must agree with the sum of the "
                    f"expected frequencies to a relative tolerance "
                    f"of {rtol}, but the percent differences are:\n"
                    f"{relative_diff}"
                )
                raise ValueError(msg)

    else:
        # Avoid warnings with the edge case of a data set with length 0
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            f_exp = np.mean(f_obs, axis=axis, keepdims=True)

    # `terms` is the array of terms that are summed along `axis` to create
    # the test statistic.  We use some specialized code for a few special
    # cases of lambda_.
    if lambda_ == 1:
        # Pearson's chi-squared statistic
        terms = (f_obs - f_exp) ** 2 / f_exp
    elif lambda_ == 0:
        # Log-likelihood ratio (i.e. G-test)
        terms = 2.0 * special.xlogy(f_obs, f_obs / f_exp)
    elif lambda_ == -1:
        # Modified log-likelihood ratio
        terms = 2.0 * special.xlogy(f_exp, f_exp / f_obs)
    else:
        # General Cressie-Read power divergence.
        terms = f_obs * ((f_obs / f_exp) ** lambda_ - 1)
        terms /= 0.5 * lambda_ * (lambda_ + 1)

    stat = np.sum(terms, axis=axis)

    num_obs = np.asarray(_count_nonmasked(terms, axis), dtype=f_obs.dtype)

    df = num_obs - 1 - ddof
    chi2 = _SimpleChi2(df)
    pvalue = _get_pvalue(stat, chi2, alternative="greater", symmetric=False)

    stat = stat[()] if stat.ndim == 0 else stat
    pvalue = pvalue[()] if pvalue.ndim == 0 else pvalue

    return Power_divergenceResult(stat, pvalue)


def rankdata(a, method="average", *, axis=None, nan_policy="propagate"):
    """Assign ranks to data, dealing with ties appropriately.

    ``method`` is one of 'average', 'min', 'max', 'dense' and 'ordinal'. Ranks start at 1.
    """
    methods = ("average", "min", "max", "dense", "ordinal")
    if method not in methods:
        raise ValueError(f'unknown method "{method}"')

    x = np.asarray(a)

    if axis is None:
        x = np.reshape(x, (-1,))
        axis = -1

    if x.size == 0:
        dtype = _result_type(x, force_floating=True)
        return np.empty_like(x, dtype=dtype)

    contains_nan = _contains_nan(x, nan_policy)

    x = np.swapaxes(x, axis, -1)
    ranks, _, _ = _rankdata(x, method)

    if contains_nan:
        i_nan = (
            np.isnan(x) if nan_policy == "omit" else np.any(np.isnan(x), axis=-1, keepdims=True)
        )
        i_nan = np.broadcast_to(i_nan, ranks.shape)
        ranks = np.where(i_nan, np.nan, ranks)

    ranks = np.swapaxes(ranks, axis, -1)
    return ranks


def _order_ranks(ranks, j):
    # Reorder ascending order `ranks` according to `j`
    ordered_ranks = np.empty(j.shape, dtype=ranks.dtype)
    np.put_along_axis(ordered_ranks, j, ranks, axis=-1)
    return ordered_ranks


def _rankdata(x, method, return_sorted=False, return_ties=False):
    # Rank data `x` by desired `method`. For methods other than 'ordinal':
    # - `return_sorted=True` ensures that the second output is sorted `x`
    # - `return_ties=True` ensures that the third output is tie data
    # Otherwise, the second/third output will be None.
    dtype = _result_type(x, force_floating=True)
    shape = x.shape

    # Get sort order
    j = np.argsort(x, axis=-1, stable=True)
    ordinal_ranks = np.broadcast_to(np.arange(1, shape[-1] + 1, dtype=dtype), shape)

    # Ordinal ranks is very easy because ties don't matter. We're done.
    if method == "ordinal":
        ranks = _order_ranks(ordinal_ranks, j)
        return (ranks, None, None)

    # Sort array
    y = np.take_along_axis(x, j, axis=-1)
    # Logical indices of unique elements
    i = np.concatenate([np.ones(shape[:-1] + (1,), dtype=bool), y[..., :-1] != y[..., 1:]], axis=-1)

    # Integer indices of unique elements
    indices = np.arange(y.size)[np.reshape(i, (-1,))]  # i gets raveled
    # Counts of unique elements
    counts = np.diff(indices, append=np.asarray([y.size], dtype=indices.dtype))

    # Compute `'min'`, `'max'`, and `'mid'` ranks of unique elements
    if method == "min":
        ranks = ordinal_ranks[i]
    elif method == "max":
        ranks = ordinal_ranks[i] + counts.astype(dtype) - 1
    elif method == "average":
        ranks = ordinal_ranks[i] + (counts.astype(dtype) - 1) / 2
    elif method == "dense":
        ranks = np.cumulative_sum(i.astype(dtype), axis=-1)[i]

    ranks = np.reshape(np.repeat(ranks, counts), shape)
    ranks = _order_ranks(ranks, j)

    t = None
    if return_ties:
        # Tie counts in sorted order: the number of appearances of the lowest element first.
        t = np.zeros(shape, dtype=dtype)
        t[i] = counts.astype(dtype)

    return ranks, y, t


LinregressResult = _make_tuple_bunch(
    "LinregressResult",
    ["slope", "intercept", "rvalue", "pvalue", "stderr"],
    extra_field_names=["intercept_stderr"],
)


def _pack_LinregressResult(slope, intercept, rvalue, pvalue, stderr, intercept_stderr):
    return LinregressResult(
        slope, intercept, rvalue, pvalue, stderr, intercept_stderr=intercept_stderr
    )


def _unpack_LinregressResult(res, _):
    return tuple(res) + (res.intercept_stderr,)


_linregress_policy = _axis_nan_policy_factory(
    _pack_LinregressResult,
    result_to_tuple=_unpack_LinregressResult,
    paired=True,
    too_small=1,
    n_outputs=6,
)


def linregress(x, y, alternative="two-sided", *, axis=0, nan_policy="propagate", keepdims=False):
    """Calculate a linear least-squares regression for two sets of measurements."""
    return _linregress_policy(
        _linregress, [x, y], {"alternative": alternative}, axis, nan_policy, keepdims
    )


def _linregress(x, y, alternative="two-sided", *, axis=0):
    x, y = _promote(x, y, force_floating=True)

    TINY = 1.0e-20

    # _axis_nan_policy decorator ensures that `axis=-1`
    n = _count_nonmasked(x, axis=-1)
    xmean = np.mean(x, axis=-1, keepdims=True)
    ymean = np.mean(y, axis=-1, keepdims=True)

    # Average sums of square differences from the mean
    #   ssxm = mean( (x-mean(x))^2 )
    #   ssxym = mean( (x-mean(x)) * (y-mean(y)) )
    x_ = _demean(x, xmean, axis=-1)
    y_ = _demean(y, ymean, axis=-1, precision_warning=False)
    xmean = np.squeeze(xmean, axis=-1)
    ymean = np.squeeze(ymean, axis=-1)

    ssxm = np.vecdot(x_, x_, axis=-1) / n
    ssym = np.vecdot(y_, y_, axis=-1) / n
    ssxym = np.vecdot(x_, y_, axis=-1) / n

    # R-value
    #   r = ssxym / sqrt( ssxm * ssym )
    degenerate = (ssxm == 0.0) | (ssym == 0.0)
    NaN = np.asarray(np.nan, dtype=np.asarray(ssxym).dtype)
    r = apply_where(
        ~degenerate,
        (ssxym, ssxm, ssym),
        lambda ssxym, ssxm, ssym: np.clip(ssxym / np.sqrt(ssxm * ssym), -1.0, 1.0),
        lambda ssxym, ssxm, ssym: np.where(ssxym == 0, NaN, 0.0),
    )

    slope = ssxym / ssxm
    intercept = ymean - slope * xmean
    with np.errstate(invalid="ignore", divide="ignore"):
        df = n - 2  # Number of degrees of freedom
        # n-2 degrees of freedom because 2 has been used up
        # to estimate the mean and standard deviation
        t = r * np.sqrt(df / ((1.0 - r + TINY) * (1.0 + r + TINY)))

        dist = _SimpleStudentT(np.asarray(df, dtype=t.dtype))
        prob = _get_pvalue(t, dist, alternative)
        prob = prob[()] if prob.ndim == 0 else prob

        slope_stderr = np.sqrt((1 - r**2) * ssym / ssxm / df)

        # Also calculate the standard error of the intercept
        # The following relationship is used:
        #   ssxm = mean( (x-mean(x))^2 )
        #        = ssx - sx*sx
        #        = mean( x^2 ) - mean(x)^2
        intercept_stderr = slope_stderr * np.sqrt(ssxm + xmean**2)

    outputs = slope, intercept, r, prob, slope_stderr, intercept_stderr
    outputs = (output[()] if np.ndim(output) == 0 else output for output in outputs)
    slope, intercept, r, prob, slope_stderr, intercept_stderr = outputs

    return LinregressResult(
        slope=slope,
        intercept=intercept,
        rvalue=r,
        pvalue=prob,
        stderr=slope_stderr,
        intercept_stderr=intercept_stderr,
    )


def _xp_mean(
    x, /, *, axis=None, weights=None, keepdims=False, nan_policy="propagate", dtype=None, warn=True
):
    """The arithmetic mean along ``axis``, optionally weighted, with SciPy's ``nan_policy``.

    Empty input gives NaN with a ``SmallSampleWarning``, and with ``nan_policy='omit'`` NaNs
    get zero weight.
    """
    x = np.asanyarray(x, dtype=dtype)
    weights = np.asarray(weights, dtype=dtype) if weights is not None else weights

    x, weights = _promote(x, weights, broadcast=True, force_floating=True)

    # handle the special case of zero-sized arrays
    message = too_small_1d_not_omit if (x.ndim == 1 or axis is None) else too_small_nd_not_omit
    if x.size == 0 or (weights is not None and weights.size == 0):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = np.mean(x, axis=axis, keepdims=keepdims)
        if warn and np.size(res) != 0:
            warnings.warn(message, SmallSampleWarning, stacklevel=2)
        return res

    contains_nan = _contains_nan(x, nan_policy)
    if weights is not None:
        contains_nan_w = _contains_nan(weights, nan_policy)
        contains_nan = contains_nan | contains_nan_w

    # Handle `nan_policy='omit'` by giving zero weight to NaNs, whether they
    # appear in `x` or `weights`. Emit warning if there is an all-NaN slice.
    if nan_policy == "omit" and contains_nan:
        nan_mask = np.isnan(x)
        if weights is not None:
            nan_mask |= np.isnan(weights)
        if warn and np.any(np.all(nan_mask, axis=axis)):
            message = too_small_1d_omit if (x.ndim == 1 or axis is None) else too_small_nd_omit
            warnings.warn(message, SmallSampleWarning, stacklevel=2)
        weights = np.ones_like(x) if weights is None else weights
        x = np.where(nan_mask, 0.0, x)
        weights = np.where(nan_mask, 0.0, weights)

    # Perform the mean calculation itself
    if weights is None:
        return np.mean(x, axis=axis, keepdims=keepdims)

    norm = np.sum(weights, axis=axis)
    wsum = np.sum(x * weights, axis=axis)
    with np.errstate(divide="ignore", invalid="ignore"):
        res = wsum / norm

    # Respect `keepdims` and convert NumPy 0-D arrays to scalars
    if keepdims:
        if axis is None:
            final_shape = (1,) * len(x.shape)
        else:
            # axis can be a scalar or sequence
            axes = (axis,) if not isinstance(axis, (tuple, list)) else axis
            final_shape = list(x.shape)
            for i in axes:
                final_shape[i] = 1

        res = np.reshape(res, tuple(final_shape))

    return res[()] if res.ndim == 0 else res


def _xp_var(x, /, *, axis=None, correction=0, keepdims=False, nan_policy="propagate", dtype=None):
    """The variance with ``correction`` delta degrees of freedom and SciPy's ``nan_policy``."""
    x = np.asanyarray(x)

    # use `_xp_mean` instead of `np.var` for desired warning behavior
    kwargs = dict(axis=axis, nan_policy=nan_policy, dtype=dtype)
    mean = _xp_mean(x, keepdims=True, **kwargs)
    x = np.asanyarray(x, dtype=mean.dtype)
    x_mean = _demean(x, mean, axis)
    x_mean_conj = np.conj(x_mean) if np.isdtype(x_mean.dtype, "complex floating") else x_mean
    var = _xp_mean(x_mean * x_mean_conj, keepdims=keepdims, **kwargs)

    if correction != 0:
        n = _count_nonmasked(x, axis, keepdims=keepdims)
        n = np.asarray(n, dtype=var.dtype)

        if nan_policy == "omit":
            nan_mask = np.isnan(x).astype(var.dtype)
            n = n - np.sum(nan_mask, axis=axis, keepdims=keepdims)

        # Produce NaNs silently when n - correction <= 0
        nc = n - correction
        factor = apply_where(nc > 0, (n, nc), operator.truediv, fill_value=np.nan)
        var *= factor

    return var[()] if var.ndim == 0 else var
