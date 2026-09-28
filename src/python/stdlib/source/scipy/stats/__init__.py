"""shellsim's ``scipy.stats``: eight plain distribution classes (no ``rv_continuous``/
``rv_discrete`` subclassing framework) plus descriptive statistics and hypothesis tests over
NumPy arrays.

Assembles the public surface from :mod:`scipy.stats._distributions` (the distribution classes),
:mod:`scipy.stats._describe` (descriptive statistics) and :mod:`scipy.stats._tests` (correlation
coefficients and hypothesis tests, including ``chi2_contingency``).
"""

from scipy.stats._describe import (
    ConstantInputWarning,
    DegenerateDataWarning,
    DescribeResult,
    ModeResult,
    SmallSampleWarning,
    describe,
    kurtosis,
    mode,
    moment,
    rankdata,
    sem,
    skew,
    trim_mean,
    zmap,
    zscore,
)
from scipy.stats._distributions import binom, chi2, expon, f, norm, poisson, t, uniform
from scipy.stats._tests import (
    chi2_contingency,
    chisquare,
    linregress,
    pearsonr,
    spearmanr,
    ttest_1samp,
    ttest_ind,
    ttest_rel,
)

__all__ = [
    "norm",
    "t",
    "chi2",
    "f",
    "uniform",
    "expon",
    "binom",
    "poisson",
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
    "pearsonr",
    "spearmanr",
    "linregress",
    "ttest_1samp",
    "ttest_ind",
    "ttest_rel",
    "chisquare",
    "chi2_contingency",
    "SmallSampleWarning",
    "ConstantInputWarning",
    "DegenerateDataWarning",
    "DescribeResult",
    "ModeResult",
]
