"""Contingency table functions, following SciPy 1.18's ``scipy/stats/contingency.py``.

shellsim implements ``margins``, ``expected_freq``, ``chi2_contingency`` and ``association``.
``crosstab``, ``relative_risk`` and ``odds_ratio`` are not implemented, and
``chi2_contingency`` rejects the resampling ``method`` argument because the resampling method
classes do not exist.
"""

import math
from functools import reduce

import numpy as np
from scipy._lib._bunch import _make_tuple_bunch
from scipy.stats._stats_py import power_divergence

__all__ = ["margins", "expected_freq", "chi2_contingency", "association"]


def margins(a):
    """Return a list of the marginal sums of the array ``a``.

    Each marginal sum keeps the dimensions of ``a``, so the sums broadcast against each other.
    """
    margsums = []
    ranged = list(range(a.ndim))
    for k in ranged:
        marg = np.apply_over_axes(np.sum, a, [j for j in ranged if j != k])
        margsums.append(marg)
    return margsums


def expected_freq(observed):
    """Compute the expected frequencies from a contingency table under independence."""
    # Typically `observed` is an integer array. If `observed` has a large
    # number of dimensions or holds large values, some of the following
    # computations may overflow, so we first switch to floating point.
    observed = np.asarray(observed, dtype=np.float64)

    # Create a list of the marginal sums.
    margsums = margins(observed)

    # Create the array of expected frequencies. The shapes of the
    # marginal sums returned by apply_over_axes() are just what we
    # need for broadcasting in the following product.
    d = observed.ndim
    expected = reduce(np.multiply, margsums) / observed.sum() ** (d - 1)
    return expected


Chi2ContingencyResult = _make_tuple_bunch(
    "Chi2ContingencyResult", ["statistic", "pvalue", "dof", "expected_freq"], []
)


def chi2_contingency(observed, correction=True, lambda_=None, *, method=None):
    """Chi-square test of independence of variables in a contingency table.

    With one degree of freedom and ``correction=True``, Yates' continuity correction moves each
    observed count 0.5 towards its expected value.
    """
    observed = np.asarray(observed)
    if np.any(observed < 0):
        raise ValueError("All values in `observed` must be nonnegative.")
    if observed.size == 0:
        raise ValueError("No data; `observed` has size 0.")

    expected = expected_freq(observed)
    if np.any(expected == 0):
        # Include one of the positions where expected is zero in
        # the exception message.
        zeropos = list(zip(*np.nonzero(expected == 0)))[0]
        raise ValueError(
            "The internally computed table of expected "
            f"frequencies has a zero element at {zeropos}."
        )

    if method is not None:
        raise NotImplementedError(
            "chi2_contingency(..., method=...) is not supported by shellsim's SciPy"
        )

    # The degrees of freedom
    dof = expected.size - sum(expected.shape) + expected.ndim - 1

    if dof == 0:
        # Degenerate case; this occurs when `observed` is 1D (or, more
        # generally, when it has only one nontrivial dimension).  In this
        # case, we also have observed == expected, so chi2 is 0.
        chi2 = 0.0
        p = 1.0
    else:
        if dof == 1 and correction:
            # Adjust `observed` according to Yates' correction for continuity.
            # Magnitude of correction no bigger than difference; see gh-13875
            diff = expected - observed
            direction = np.sign(diff)
            magnitude = np.minimum(0.5, np.abs(diff))
            observed = observed + magnitude * direction

        chi2, p = power_divergence(
            observed, expected, ddof=observed.size - 1 - dof, axis=None, lambda_=lambda_
        )

    return Chi2ContingencyResult(chi2, p, dof, expected)


def association(observed, method="cramer", correction=False, lambda_=None):
    """Calculate the degree of association between two nominal variables.

    ``method`` is 'cramer' (Cramér's V), 'tschuprow' (Tschuprow's T) or 'pearson' (Pearson's
    contingency coefficient).
    """
    arr = np.asarray(observed)
    if not np.issubdtype(arr.dtype, np.integer):
        raise ValueError("`observed` must be an integer array.")

    if len(arr.shape) != 2:
        raise ValueError("method only accepts 2d arrays")

    chi2_stat = chi2_contingency(arr, correction=correction, lambda_=lambda_)

    phi2 = chi2_stat.statistic / arr.sum()
    n_rows, n_cols = arr.shape
    if method == "cramer":
        value = phi2 / min(n_cols - 1, n_rows - 1)
    elif method == "tschuprow":
        value = phi2 / math.sqrt((n_rows - 1) * (n_cols - 1))
    elif method == "pearson":
        value = phi2 / (1 + phi2)
    else:
        raise ValueError(
            "Invalid argument value: 'method' argument must "
            "be 'cramer', 'tschuprow', or 'pearson'"
        )

    return math.sqrt(value)
