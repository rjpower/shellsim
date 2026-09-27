"""shellsim's ``scipy.stats``.

The implemented surface follows SciPy 1.18:

- the ``rv_continuous`` and ``rv_discrete`` frameworks, with ``norm``, ``t``, ``chi2``, ``f``,
  ``uniform`` and ``expon``, and the discrete ``binom`` and ``poisson``;
- summary statistics: ``describe``, ``moment``, ``skew``, ``kurtosis``, ``mode``, ``sem``,
  ``zscore``, ``zmap``, ``trim_mean``, ``rankdata`` and ``entropy``;
- correlation and regression: ``pearsonr``, ``spearmanr`` and ``linregress``;
- tests: ``ttest_1samp``, ``ttest_ind``, ``ttest_ind_from_stats``, ``ttest_rel``,
  ``chisquare``, ``power_divergence`` and ``chi2_contingency``.

Accessing any other name in SciPy's ``scipy.stats.__all__`` raises ``NotImplementedError``.
"""

import scipy.stats.contingency as contingency
from scipy.stats._continuous_distns import chi2, expon, f, norm, t, uniform
from scipy.stats._discrete_distns import binom, poisson
from scipy.stats._distn_infrastructure import rv_continuous, rv_discrete
from scipy.stats._entropy import entropy
from scipy.stats._stats_py import (
    chisquare,
    describe,
    kurtosis,
    linregress,
    mode,
    moment,
    pearsonr,
    power_divergence,
    rankdata,
    sem,
    skew,
    spearmanr,
    trim_mean,
    ttest_1samp,
    ttest_ind,
    ttest_ind_from_stats,
    ttest_rel,
    zmap,
    zscore,
)
from scipy.stats._warnings_errors import (
    ConstantInputWarning,
    DegenerateDataWarning,
    FitError,
    NearConstantInputWarning,
)
from scipy.stats.contingency import chi2_contingency

__all__ = [
    "ConstantInputWarning",
    "DegenerateDataWarning",
    "FitError",
    "NearConstantInputWarning",
    "binom",
    "chi2",
    "chi2_contingency",
    "chisquare",
    "contingency",
    "describe",
    "entropy",
    "expon",
    "f",
    "kurtosis",
    "linregress",
    "mode",
    "moment",
    "norm",
    "pearsonr",
    "poisson",
    "power_divergence",
    "rankdata",
    "rv_continuous",
    "rv_discrete",
    "sem",
    "skew",
    "spearmanr",
    "t",
    "trim_mean",
    "ttest_1samp",
    "ttest_ind",
    "ttest_ind_from_stats",
    "ttest_rel",
    "uniform",
    "zmap",
    "zscore",
]

# The rest of SciPy 1.18's `scipy.stats.__all__`.
_UNSUPPORTED = frozenset(
    (
        'Binomial', 'BootstrapMethod', 'CensoredData', 'Covariance', 'Logistic', 'Mixture',
        'MonteCarloMethod', 'Normal', 'PermutationMethod', 'Uniform', 'abs', 'alexandergovern',
        'alpha', 'anderson', 'anderson_ksamp', 'anglit', 'ansari', 'arcsine', 'argus',
        'barnard_exact', 'bartlett', 'bayes_mvs', 'bernoulli', 'beta', 'betabinom', 'betanbinom',
        'betaprime', 'biasedurn', 'binned_statistic', 'binned_statistic_2d', 'binned_statistic_dd',
        'binomtest', 'boltzmann', 'bootstrap', 'boschloo_exact', 'boxcox', 'boxcox_llf',
        'boxcox_normmax', 'boxcox_normplot', 'bradford', 'brunnermunzel', 'burr', 'burr12',
        'bws_test', 'cauchy', 'chatterjeexi', 'chi', 'circmean', 'circstd', 'circvar',
        'combine_pvalues', 'cosine', 'cramervonmises', 'cramervonmises_2samp', 'crystalball',
        'cumfreq', 'dgamma', 'differential_entropy', 'directional_stats', 'dirichlet',
        'dirichlet_multinomial', 'distributions', 'dlaplace', 'dpareto_lognorm', 'dunnett',
        'dweibull', 'ecdf', 'energy_distance', 'epps_singleton_2samp', 'erlang', 'estimated_cdf',
        'exp', 'expectile', 'exponnorm', 'exponpow', 'exponweib', 'f_oneway',
        'false_discovery_control', 'fatiguelife', 'fisher_exact', 'fisk', 'fit', 'fligner',
        'foldcauchy', 'foldnorm', 'friedmanchisquare', 'gamma', 'gausshyper', 'gaussian_kde',
        'genexpon', 'genextreme', 'gengamma', 'genhalflogistic', 'genhyperbolic', 'geninvgauss',
        'genlogistic', 'gennorm', 'genpareto', 'geom', 'gibrat', 'gmean', 'gompertz',
        'goodness_of_fit', 'gstd', 'gumbel_l', 'gumbel_r', 'gzscore', 'halfcauchy', 'halfgennorm',
        'halflogistic', 'halfnorm', 'hmean', 'hypergeom', 'hypsecant', 'invgamma', 'invgauss',
        'invweibull', 'invwishart', 'iqr', 'irwinhall', 'jarque_bera', 'jf_skew_t', 'johnsonsb',
        'johnsonsu', 'kappa3', 'kappa4', 'kde', 'kendalltau', 'kruskal', 'ks_1samp', 'ks_2samp',
        'ksone', 'kstat', 'kstatvar', 'kstest', 'kstwo', 'kstwobign', 'kurtosistest', 'landau',
        'laplace', 'laplace_asymmetric', 'levene', 'levy', 'levy_l', 'levy_stable', 'lmoment',
        'log', 'loggamma', 'logistic', 'loglaplace', 'lognorm', 'logrank', 'logser', 'loguniform',
        'lomax', 'make_distribution', 'mannwhitneyu', 'matrix_normal', 'matrix_t', 'maxwell',
        'median_abs_deviation', 'median_test', 'mielke', 'monte_carlo_test', 'mood', 'morestats',
        'moyal', 'mstats', 'mstats_basic', 'mstats_extras', 'multinomial', 'multiscale_graphcorr',
        'multivariate_hypergeom', 'multivariate_normal', 'multivariate_t', 'mvn', 'mvsdist',
        'nakagami', 'nbinom', 'ncf', 'nchypergeom_fisher', 'nchypergeom_wallenius', 'nct', 'ncx2',
        'nhypergeom', 'normal_inverse_gamma', 'normaltest', 'norminvgauss', 'obrientransform',
        'order_statistic', 'ortho_group', 'page_trend_test', 'pareto', 'pearson3',
        'percentileofscore', 'permutation_test', 'planck', 'pmean', 'pointbiserialr',
        'poisson_binom', 'poisson_means_test', 'power', 'powerlaw', 'powerlognorm', 'powernorm',
        'ppcc_max', 'ppcc_plot', 'probplot', 'qmc', 'quantile', 'quantile_test', 'randint',
        'random_correlation', 'random_table', 'ranksums', 'rayleigh', 'rdist', 'recipinvgauss',
        'reciprocal', 'rel_breitwigner', 'relfreq', 'rice', 'rv_histogram', 'scoreatpercentile',
        'semicircular', 'shapiro', 'siegelslopes', 'sigmaclip', 'skellam', 'skewcauchy', 'skewnorm',
        'skewtest', 'sobol_indices', 'somersd', 'spearmanrho', 'special_ortho_group', 'stats',
        'studentized_range', 'theilslopes', 'tiecorrect', 'tmax', 'tmean', 'tmin', 'trapezoid',
        'triang', 'trim1', 'trimboth', 'truncate', 'truncexpon', 'truncnorm', 'truncpareto',
        'truncweibull_min', 'tsem', 'tstd', 'tukey_hsd', 'tukeylambda', 'tvar', 'uniform_direction',
        'unitary_group', 'variation', 'vonmises', 'vonmises_fisher', 'vonmises_line', 'wald',
        'wasserstein_distance', 'wasserstein_distance_nd', 'weibull_max', 'weibull_min',
        'weightedtau', 'wilcoxon', 'wishart', 'wrapcauchy', 'yeojohnson', 'yeojohnson_llf',
        'yeojohnson_normmax', 'yeojohnson_normplot', 'yulesimon', 'zipf', 'zipfian',
    )
)


def __getattr__(name):
    if name in _UNSUPPORTED:
        raise NotImplementedError(f"scipy.stats.{name} is not supported by shellsim's SciPy")
    raise AttributeError(f"module 'scipy.stats' has no attribute '{name}'")
