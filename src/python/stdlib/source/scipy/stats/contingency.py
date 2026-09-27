"""Contingency table statistics: ``chi2_contingency``, ``margins``, ``expected_freq`` and
``association``.

All are elementary: ``expected_freq`` is an outer product of the table's margins divided by the
total, ``chi2_contingency`` is a Pearson chi-squared test (with Yates' continuity correction for
a 2x2 table) against that expectation, and ``association`` rescales its chi-squared statistic.
"""

from collections import namedtuple

import numpy as np

from scipy import special
from scipy.stats._stats import _scalarize

__all__ = ["chi2_contingency", "margins", "expected_freq", "association"]


Chi2ContingencyResult = namedtuple(
    "Chi2ContingencyResult", ["statistic", "pvalue", "dof", "expected_freq"]
)


def margins(a):
    """The sum of `a` over every axis but one, for each axis, keeping that axis's dimension."""
    a = np.asarray(a)
    return [np.sum(a, axis=tuple(i for i in range(a.ndim) if i != k), keepdims=True) for k in range(a.ndim)]


def expected_freq(observed):
    """The independence-model expected counts: the outer product of `observed`'s margins."""
    observed = np.asarray(observed, dtype=float)
    total = observed.sum()
    if total == 0:
        return np.zeros_like(observed)
    expected = np.ones_like(observed)
    for margin in margins(observed):
        expected = expected * margin
    return expected / total ** (observed.ndim - 1)


def chi2_contingency(observed, correction=True, lambda_=None, method=None):
    if method is not None:
        raise NotImplementedError("chi2_contingency(..., method=...) is not supported by shellsim's SciPy")
    observed = np.asarray(observed, dtype=float)
    expected = expected_freq(observed)
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
    pvalue = float(special.chdtrc(dof, statistic))
    return Chi2ContingencyResult(statistic, pvalue, dof, expected)


def association(observed, method="cramer", correction=False, lambda_=None):
    observed = np.asarray(observed, dtype=float)
    statistic = chi2_contingency(observed, correction=correction, lambda_=lambda_).statistic
    n = observed.sum()
    r, c = observed.shape
    if method == "cramer":
        value = np.sqrt(statistic / (n * min(r - 1, c - 1)))
    elif method == "tschuprow":
        value = np.sqrt(statistic / (n * np.sqrt((r - 1) * (c - 1))))
    elif method == "pearson":
        value = np.sqrt(statistic / (statistic + n))
    else:
        raise ValueError("Invalid argument value: 'method' argument must be 'cramer', 'tschuprow', or 'pearson'")
    return _scalarize(value)
