"""Correlation coefficients and hypothesis tests: ``pearsonr``, ``spearmanr``, ``linregress``,
the Student's t-tests, ``chisquare`` and ``chi2_contingency``.

Every p-value here reduces to the Student's t or chi-squared distribution function, computed
through the same ``scipy.special`` incomplete gamma/beta ufuncs ``scipy.stats``'s own
distributions use (``scipy.stats._distributions._t_cdf`` and ``special.gammaincc``), so results
agree with SciPy through the same floating-point path documented in docs/scipy.md.
"""

import math
import warnings
from collections import namedtuple

import numpy as np

from scipy import special
from scipy.stats._describe import ConstantInputWarning, _rankdata_1d, _scalarize, _tuple_bunch
from scipy.stats._distributions import _t_cdf

__all__ = [
    "pearsonr",
    "spearmanr",
    "linregress",
    "ttest_1samp",
    "ttest_ind",
    "ttest_rel",
    "chisquare",
    "chi2_contingency",
]


SignificanceResult = namedtuple("SignificanceResult", ["statistic", "pvalue"])
Chi2ContingencyResult = namedtuple("Chi2ContingencyResult", ["statistic", "pvalue", "dof", "expected_freq"])
LinregressResult = _tuple_bunch(
    "LinregressResult",
    ["slope", "intercept", "rvalue", "pvalue", "stderr"],
    ["intercept_stderr"],
)
TtestResult = _tuple_bunch("TtestResult", ["statistic", "pvalue"], ["df"])


def _ttest_result(statistic, pvalue, df):
    return TtestResult(statistic, pvalue, df=df)


def _broadcast_df(df, like):
    return _scalarize(np.broadcast_to(np.asarray(df, dtype=float), np.shape(np.asarray(like))).copy())


def _t_pvalue(t, df, alternative):
    t = np.asarray(t, dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        if alternative == "two-sided":
            return 2.0 * _t_cdf(-np.abs(t), df)
        if alternative == "less":
            return _t_cdf(t, df)
        if alternative == "greater":
            return _t_cdf(-t, df)
    raise ValueError("alternative must be 'less', 'greater' or 'two-sided'")


def _pearson_core(x, y, axis, alternative):
    mx = np.mean(x, axis=axis, keepdims=True)
    my = np.mean(y, axis=axis, keepdims=True)
    dx = x - mx
    dy = y - my
    sxx = np.sum(dx * dx, axis=axis)
    syy = np.sum(dy * dy, axis=axis)
    sxy = np.sum(dx * dy, axis=axis)
    den = np.sqrt(sxx * syy)
    constant = den == 0
    with np.errstate(invalid="ignore", divide="ignore"):
        r = np.clip(sxy / den, -1.0, 1.0)
    df = x.shape[axis] - 2
    with np.errstate(invalid="ignore", divide="ignore"):
        t = r * np.sqrt(df / (1.0 - r * r))
    p = _t_pvalue(t, df, alternative)
    r = np.where(constant, np.nan, r)
    p = np.where(constant, np.nan, p)
    return r, p, constant


def _warn_constant():
    warnings.warn(
        "An input array is constant; the correlation coefficient is not defined.",
        ConstantInputWarning,
        stacklevel=3,
    )


def pearsonr(x, y, *, alternative="two-sided", axis=0):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.shape[axis] < 2:
        raise ValueError("`x` and `y` must have length at least 2.")
    r, p, constant = _pearson_core(x, y, axis, alternative)
    if np.any(constant):
        _warn_constant()
    return SignificanceResult(_scalarize(r), _scalarize(p))


def spearmanr(a, b=None, axis=0, alternative="two-sided"):
    a = np.asarray(a, dtype=float)

    def ranked(arr, axis):
        return np.apply_along_axis(lambda row: _rankdata_1d(row, "average"), axis, arr)

    if b is not None:
        b = np.asarray(b, dtype=float)
        r, p, constant = _pearson_core(ranked(a, axis), ranked(b, axis), axis, alternative)
        if np.any(constant):
            _warn_constant()
        return SignificanceResult(_scalarize(r), _scalarize(p))

    # A single 2-D array with no `b` correlates every pair of variables (the axis opposite
    # `axis`), like `numpy.corrcoef`, rather than reducing to one statistic.
    variables = a if axis == 1 else a.T
    ranks = ranked(variables, 1)
    n = ranks.shape[1]
    stat = np.corrcoef(ranks)
    df = n - 2
    with np.errstate(invalid="ignore", divide="ignore"):
        t = stat * np.sqrt(df / (1.0 - stat * stat))
    pvalue = _t_pvalue(t, df, alternative)
    diag = np.arange(stat.shape[0])
    stat[diag, diag] = 1.0
    pvalue[diag, diag] = 0.0
    return SignificanceResult(stat, pvalue)


def linregress(x, y=None, alternative="two-sided"):
    x = np.asarray(x, dtype=float)
    if y is None:
        y = x[:, 1]
        x = x[:, 0]
    else:
        y = np.asarray(y, dtype=float)
    n = x.shape[0]
    mx, my = np.mean(x), np.mean(y)
    dx, dy = x - mx, y - my
    sxx, syy, sxy = np.sum(dx * dx), np.sum(dy * dy), np.sum(dx * dy)
    slope = sxy / sxx
    intercept = my - slope * mx
    r = min(max(sxy / math.sqrt(sxx * syy), -1.0), 1.0)
    df = n - 2
    t = r * math.sqrt(df / (1.0 - r * r))
    pvalue = float(_t_pvalue(t, df, alternative))
    residuals = y - (intercept + slope * x)
    stderr = math.sqrt(np.sum(residuals**2) / df) / math.sqrt(sxx)
    intercept_stderr = stderr * math.sqrt(np.sum(x * x) / n)
    return LinregressResult(slope, intercept, r, pvalue, stderr, intercept_stderr=intercept_stderr)


def ttest_1samp(a, popmean, axis=0, alternative="two-sided"):
    a = np.asarray(a, dtype=float)
    popmean = np.asarray(popmean, dtype=float)
    n = a.shape[axis]
    mean = np.mean(a, axis=axis)
    se = np.std(a, axis=axis, ddof=1) / math.sqrt(n)
    with np.errstate(invalid="ignore", divide="ignore"):
        t = (mean - popmean) / se
    df = n - 1
    p = _t_pvalue(t, df, alternative)
    return _ttest_result(_scalarize(t), _scalarize(p), _broadcast_df(df, t))


def ttest_rel(a, b, axis=0, alternative="two-sided"):
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    n = d.shape[axis]
    mean = np.mean(d, axis=axis)
    se = np.std(d, axis=axis, ddof=1) / math.sqrt(n)
    with np.errstate(invalid="ignore", divide="ignore"):
        t = mean / se
    df = n - 1
    p = _t_pvalue(t, df, alternative)
    return _ttest_result(_scalarize(t), _scalarize(p), _broadcast_df(df, t))


def _winsorized_stats(x, axis, trim):
    x = np.moveaxis(x, axis, -1)
    n = x.shape[-1]
    cut = int(math.floor(n * trim))
    sorted_x = np.sort(x, axis=-1)
    trimmed_mean = np.mean(sorted_x[..., cut : n - cut], axis=-1)
    if cut > 0:
        lo = sorted_x[..., cut : cut + 1]
        hi = sorted_x[..., n - 1 - cut : n - cut]
        winsorized = np.clip(x, lo, hi)
    else:
        winsorized = x
    h = n - 2 * cut
    winsorized_var = np.var(winsorized, axis=-1, ddof=1)
    d = (n - 1) * winsorized_var / (h * (h - 1))
    return trimmed_mean, d, h


def ttest_ind(a, b, axis=0, equal_var=True, alternative="two-sided", trim=0):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if trim > 0:
        # Yuen's trimmed (Welch-style) t-test: a per-group Winsorized variance in the standard
        # error, but SciPy still pools the degrees of freedom as `h1 + h2 - 2`.
        tm1, d1, h1 = _winsorized_stats(a, axis, trim)
        tm2, d2, h2 = _winsorized_stats(b, axis, trim)
        se = np.sqrt(d1 + d2)
        diff = tm1 - tm2
        df = h1 + h2 - 2
    else:
        n1, n2 = a.shape[axis], b.shape[axis]
        m1, m2 = np.mean(a, axis=axis), np.mean(b, axis=axis)
        v1, v2 = np.var(a, axis=axis, ddof=1), np.var(b, axis=axis, ddof=1)
        diff = m1 - m2
        if equal_var:
            df = n1 + n2 - 2
            pooled = ((n1 - 1) * v1 + (n2 - 1) * v2) / df
            se = np.sqrt(pooled * (1.0 / n1 + 1.0 / n2))
        else:
            se2 = v1 / n1 + v2 / n2
            se = np.sqrt(se2)
            with np.errstate(invalid="ignore", divide="ignore"):
                df = se2**2 / ((v1 / n1) ** 2 / (n1 - 1) + (v2 / n2) ** 2 / (n2 - 1))
    with np.errstate(invalid="ignore", divide="ignore"):
        t = diff / se
    p = _t_pvalue(t, df, alternative)
    return _ttest_result(_scalarize(t), _scalarize(p), _broadcast_df(df, t))


def chisquare(f_obs, f_exp=None, ddof=0, axis=0):
    """Pearson's chi-square goodness-of-fit test: `sum((observed - expected)^2 / expected)`
    against the chi-square distribution with `n - 1 - ddof` degrees of freedom."""
    f_obs = np.asarray(f_obs, dtype=float)
    n = f_obs.shape[axis]
    if f_exp is None:
        f_exp = np.broadcast_to(np.mean(f_obs, axis=axis, keepdims=True), f_obs.shape)
    else:
        f_exp = np.asarray(f_exp, dtype=float)
        if not np.allclose(np.sum(f_obs, axis=axis), np.sum(f_exp, axis=axis), rtol=1e-8):
            raise ValueError(
                "For each axis slice, the sum of the observed frequencies must agree with the "
                "sum of the expected frequencies to a relative tolerance of 1e-8."
            )
    stat = np.sum((f_obs - f_exp) ** 2 / f_exp, axis=axis)
    df = n - 1 - ddof
    p = special.gammaincc(df / 2.0, stat / 2.0)
    return SignificanceResult(_scalarize(stat), _scalarize(p))


def _expected_freq(observed):
    """The independence-model expected counts: the outer product of `observed`'s margins."""
    observed = np.asarray(observed, dtype=float)
    total = observed.sum()
    if total == 0:
        return np.zeros_like(observed)
    expected = np.ones_like(observed)
    for axis in range(observed.ndim):
        margin = np.sum(observed, axis=tuple(i for i in range(observed.ndim) if i != axis), keepdims=True)
        expected = expected * margin
    return expected / total ** (observed.ndim - 1)


def chi2_contingency(observed, correction=True, lambda_=None, method=None):
    """Pearson's chi-square test of independence, with Yates' continuity correction for a 2x2
    table by default."""
    if method is not None:
        raise NotImplementedError("chi2_contingency(..., method=...) is not supported by shellsim's SciPy")
    observed = np.asarray(observed, dtype=float)
    expected = _expected_freq(observed)
    if np.any(expected == 0):
        index = tuple(int(i) for i in np.argwhere(expected == 0)[0])
        raise ValueError(f"The internally computed table of expected frequencies has a zero element at {index}.")
    dof = expected.size - sum(expected.shape) + expected.ndim - 1
    if dof == 0:
        return Chi2ContingencyResult(0.0, 1.0, 0, expected)
    diff = observed - expected
    if dof == 1 and correction:
        diff = np.sign(diff) * np.clip(np.abs(diff) - 0.5, 0.0, None)
    statistic = float(np.sum(diff**2 / expected))
    pvalue = float(special.gammaincc(dof / 2.0, statistic / 2.0))
    return Chi2ContingencyResult(statistic, pvalue, dof, expected)
