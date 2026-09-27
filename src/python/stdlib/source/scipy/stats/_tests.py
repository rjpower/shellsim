"""Correlation coefficients and hypothesis tests: ``pearsonr``, ``spearmanr``, ``linregress``,
the Student's t-tests and the chi-square/power-divergence family.

Every p-value here reduces to the Student's t or chi-squared distribution function, computed
through :mod:`scipy.special` (``stdtr``/``stdtrit`` and ``chdtrc``) exactly as the corresponding
``scipy.stats`` distributions do, so results agree with SciPy through the same floating-point
path documented in docs/scipy.md.
"""

import math
import warnings

import numpy as np

from scipy import special
from scipy.stats._stats import ConstantInputWarning, _Result, _rankdata_1d, _scalarize

__all__ = [
    "pearsonr",
    "spearmanr",
    "linregress",
    "ttest_1samp",
    "ttest_ind",
    "ttest_ind_from_stats",
    "ttest_rel",
    "chisquare",
    "power_divergence",
]


class SignificanceResult(_Result):
    _fields = ("statistic", "pvalue")


class Power_divergenceResult(_Result):
    _fields = ("statistic", "pvalue")


class Ttest_indResult(_Result):
    _fields = ("statistic", "pvalue")


class LinregressResult(_Result):
    _fields = ("slope", "intercept", "rvalue", "pvalue", "stderr")

    def __init__(self, slope, intercept, rvalue, pvalue, stderr, intercept_stderr):
        super().__init__(slope, intercept, rvalue, pvalue, stderr)
        self.intercept_stderr = intercept_stderr


class PearsonRResult(_Result):
    _fields = ("statistic", "pvalue")

    def __init__(self, statistic, pvalue, n, alternative):
        super().__init__(statistic, pvalue)
        self._n = n
        self._alternative = alternative

    @property
    def correlation(self):
        return self.statistic

    def confidence_interval(self, confidence_level=0.95, method=None):
        r = np.asarray(self.statistic, dtype=float)
        se = 1.0 / math.sqrt(self._n - 3)
        with np.errstate(invalid="ignore", divide="ignore"):
            z = np.arctanh(r)
        if self._alternative == "two-sided":
            crit = special.ndtri(0.5 + confidence_level / 2.0)
            lo, hi = z - crit * se, z + crit * se
        elif self._alternative == "less":
            crit = special.ndtri(confidence_level)
            lo, hi = -np.inf, z + crit * se
        else:
            crit = special.ndtri(confidence_level)
            lo, hi = z - crit * se, np.inf
        # tanh saturates to +/-1 at infinite bounds, so a one-sided interval needs no special case.
        return _scalarize(np.tanh(lo)), _scalarize(np.tanh(hi))


class TtestResult(_Result):
    _fields = ("statistic", "pvalue", "df")

    def __init__(self, statistic, pvalue, df, *, center, standard_error, alternative):
        super().__init__(statistic, pvalue, df)
        self._center = center
        self._se = standard_error
        self._alternative = alternative

    def confidence_interval(self, confidence_level=0.95):
        df = np.asarray(self.df, dtype=float)
        se = np.asarray(self._se, dtype=float)
        center = np.asarray(self._center, dtype=float)
        if self._alternative == "two-sided":
            crit = special.stdtrit(df, 0.5 + confidence_level / 2.0)
            lo, hi = center - crit * se, center + crit * se
        elif self._alternative == "less":
            crit = special.stdtrit(df, confidence_level)
            lo, hi = np.full(center.shape, -np.inf) if center.ndim else -np.inf, center + crit * se
        else:
            crit = special.stdtrit(df, confidence_level)
            lo, hi = center - crit * se, np.full(center.shape, np.inf) if center.ndim else np.inf
        return _scalarize(np.asarray(lo)), _scalarize(np.asarray(hi))


def _broadcast_df(df, like):
    return _scalarize(np.broadcast_to(np.asarray(df, dtype=float), np.shape(np.asarray(like))).copy())


def _t_pvalue(t, df, alternative):
    t = np.asarray(t, dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        if alternative == "two-sided":
            return 2.0 * special.stdtr(df, -np.abs(t))
        if alternative == "less":
            return special.stdtr(df, t)
        if alternative == "greater":
            return special.stdtr(df, -t)
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
    return PearsonRResult(_scalarize(r), _scalarize(p), n=x.shape[axis], alternative=alternative)


def spearmanr(a, b=None, axis=0, nan_policy="propagate", alternative="two-sided"):
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
    return LinregressResult(slope, intercept, r, pvalue, stderr, intercept_stderr)


def ttest_1samp(a, popmean, axis=0, nan_policy="propagate", alternative="two-sided"):
    a = np.asarray(a, dtype=float)
    popmean = np.asarray(popmean, dtype=float)
    n = a.shape[axis]
    mean = np.mean(a, axis=axis)
    se = np.std(a, axis=axis, ddof=1) / math.sqrt(n)
    with np.errstate(invalid="ignore", divide="ignore"):
        t = (mean - popmean) / se
    df = n - 1
    p = _t_pvalue(t, df, alternative)
    return TtestResult(
        _scalarize(t), _scalarize(p), _broadcast_df(df, t),
        center=_scalarize(mean), standard_error=_scalarize(se), alternative=alternative,
    )


def ttest_rel(a, b, axis=0, nan_policy="propagate", alternative="two-sided"):
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    n = d.shape[axis]
    mean = np.mean(d, axis=axis)
    se = np.std(d, axis=axis, ddof=1) / math.sqrt(n)
    with np.errstate(invalid="ignore", divide="ignore"):
        t = mean / se
    df = n - 1
    p = _t_pvalue(t, df, alternative)
    return TtestResult(
        _scalarize(t), _scalarize(p), _broadcast_df(df, t),
        center=_scalarize(mean), standard_error=_scalarize(se), alternative=alternative,
    )


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


def ttest_ind(a, b, axis=0, equal_var=True, nan_policy="propagate", alternative="two-sided", trim=0):
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
    return TtestResult(
        _scalarize(t), _scalarize(p), _broadcast_df(df, t),
        center=_scalarize(diff), standard_error=_scalarize(se), alternative=alternative,
    )


def ttest_ind_from_stats(mean1, std1, nobs1, mean2, std2, nobs2, equal_var=True, alternative="two-sided"):
    v1, v2 = std1**2, std2**2
    if equal_var:
        df = nobs1 + nobs2 - 2
        pooled = ((nobs1 - 1) * v1 + (nobs2 - 1) * v2) / df
        se = math.sqrt(pooled * (1.0 / nobs1 + 1.0 / nobs2))
    else:
        se2 = v1 / nobs1 + v2 / nobs2
        se = math.sqrt(se2)
        df = se2**2 / ((v1 / nobs1) ** 2 / (nobs1 - 1) + (v2 / nobs2) ** 2 / (nobs2 - 1))
    diff = mean1 - mean2
    t = diff / se
    p = float(_t_pvalue(t, df, alternative))
    return Ttest_indResult(t, p)


_LAMBDA_NAMES = {
    "pearson": 1.0,
    "log-likelihood": 0.0,
    "freeman-tukey": -0.5,
    "mod-log-likelihood": -1.0,
    "neyman": -2.0,
    "cressie-read": 2.0 / 3.0,
}


def power_divergence(f_obs, f_exp=None, ddof=0, axis=0, lambda_=None):
    f_obs = np.asarray(f_obs, dtype=float)
    if lambda_ is None:
        lambda_ = 1.0
    elif isinstance(lambda_, str):
        if lambda_ not in _LAMBDA_NAMES:
            raise ValueError(f"invalid string for lambda_: {lambda_!r}")
        lambda_ = _LAMBDA_NAMES[lambda_]
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
    with np.errstate(invalid="ignore", divide="ignore"):
        if lambda_ == 0.0:
            terms = np.where(f_obs == 0, 0.0, f_obs * np.log(f_obs / f_exp))
            stat = 2.0 * np.sum(terms, axis=axis)
        elif lambda_ == -1.0:
            terms = np.where(f_exp == 0, 0.0, f_exp * np.log(f_exp / f_obs))
            stat = 2.0 * np.sum(terms, axis=axis)
        else:
            terms = f_obs * ((f_obs / f_exp) ** lambda_ - 1.0)
            stat = 2.0 / (lambda_ * (lambda_ + 1.0)) * np.sum(terms, axis=axis)
    df = n - 1 - ddof
    p = special.chdtrc(df, stat)
    return Power_divergenceResult(_scalarize(stat), _scalarize(p))


def chisquare(f_obs, f_exp=None, ddof=0, axis=0):
    return power_divergence(f_obs, f_exp, ddof=ddof, axis=axis, lambda_="pearson")
