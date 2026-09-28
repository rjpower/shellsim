"""shellsim's ``scipy.stats``.

Assembles the public surface from :mod:`scipy.stats._distributions` (the ``rv_continuous``/
``rv_discrete`` framework and the built-in distributions), :mod:`scipy.stats._stats` (summary
statistics), :mod:`scipy.stats._tests` (correlation coefficients and hypothesis tests) and
:mod:`scipy.stats.contingency`.
"""

from scipy.stats import contingency
from scipy.stats._distributions import binom, chi2, expon, f, norm, poisson, rv_continuous, rv_discrete, t, uniform
from scipy.stats._stats import (
    ConstantInputWarning,
    DegenerateDataWarning,
    DescribeResult,
    ModeResult,
    SmallSampleWarning,
    describe,
    entropy,
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
from scipy.stats._tests import (
    chisquare,
    linregress,
    pearsonr,
    power_divergence,
    spearmanr,
    ttest_1samp,
    ttest_ind,
    ttest_ind_from_stats,
    ttest_rel,
)
from scipy.stats.contingency import chi2_contingency

__all__ = [
    "rv_continuous",
    "rv_discrete",
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
    "entropy",
    "pearsonr",
    "spearmanr",
    "linregress",
    "ttest_1samp",
    "ttest_ind",
    "ttest_ind_from_stats",
    "ttest_rel",
    "chisquare",
    "power_divergence",
    "chi2_contingency",
    "contingency",
    "SmallSampleWarning",
    "ConstantInputWarning",
    "DegenerateDataWarning",
    "DescribeResult",
    "ModeResult",
]
