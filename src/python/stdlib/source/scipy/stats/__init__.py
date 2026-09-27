"""shellsim's ``scipy.stats``.

Assembles the public surface from :mod:`scipy.stats._distributions` (the ``rv_continuous``/
``rv_discrete`` framework and the built-in distributions), :mod:`scipy.stats._stats` (summary
statistics), :mod:`scipy.stats._tests` (correlation coefficients and hypothesis tests) and
:mod:`scipy.stats.contingency`. Every other name in real SciPy's ``scipy.stats.__all__`` resolves
through ``__getattr__`` to ``NotImplementedError``, so a program that only reads that message
gets the same signal it would from a plain missing feature; a name outside that list raises the
usual ``AttributeError``.
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

# The rest of real SciPy's `scipy.stats.__all__`: distributions, hypothesis tests and helpers
# shellsim does not implement. Accessing one raises `NotImplementedError` instead of the plain
# `AttributeError` an unrecognized name gets, so a caller can tell "not supported" from "not
# SciPy".
_UNSUPPORTED_NAMES = frozenset(
    {
        "Binomial", "BootstrapMethod", "CensoredData", "Covariance", "FitError", "Logistic",
        "Mixture", "MonteCarloMethod", "NearConstantInputWarning", "Normal", "PermutationMethod",
        "Uniform", "abs", "alexandergovern", "alpha", "anderson", "anderson_ksamp", "anglit",
        "ansari", "arcsine", "argus", "barnard_exact", "bartlett", "bayes_mvs", "bernoulli",
        "beta", "betabinom", "betanbinom", "betaprime", "biasedurn", "binned_statistic",
        "binned_statistic_2d", "binned_statistic_dd", "binomtest", "boltzmann", "bootstrap",
        "boschloo_exact", "boxcox", "boxcox_llf", "boxcox_normmax", "boxcox_normplot", "bradford",
        "brunnermunzel", "burr", "burr12", "bws_test", "cauchy", "chatterjeexi", "chi", "circmean",
        "circstd", "circvar", "combine_pvalues", "cosine", "cramervonmises", "cramervonmises_2samp",
        "crystalball", "cumfreq", "dgamma", "differential_entropy", "directional_stats",
        "dirichlet", "dirichlet_multinomial", "distributions", "dlaplace", "dpareto_lognorm",
        "dunnett", "dweibull", "ecdf", "energy_distance", "epps_singleton_2samp", "erlang",
        "estimated_cdf", "exp", "expectile", "exponnorm", "exponpow", "exponweib", "f_oneway",
        "false_discovery_control", "fatiguelife", "fisher_exact", "fisk", "fit", "fligner",
        "foldcauchy", "foldnorm", "friedmanchisquare", "gamma", "gausshyper", "gaussian_kde",
        "genexpon", "genextreme", "gengamma", "genhalflogistic", "genhyperbolic", "geninvgauss",
        "genlogistic", "gennorm", "genpareto", "geom", "gibrat", "gmean", "gompertz",
        "goodness_of_fit", "gstd", "gumbel_l", "gumbel_r", "gzscore", "halfcauchy", "halfgennorm",
        "halflogistic", "halfnorm", "hmean", "hypergeom", "hypsecant", "invgamma", "invgauss",
        "invweibull", "invwishart", "iqr", "irwinhall", "jarque_bera", "jf_skew_t", "johnsonsb",
        "johnsonsu", "kappa3", "kappa4", "kde", "kendalltau", "kruskal", "ks_1samp", "ks_2samp",
        "ksone", "kstat", "kstatvar", "kstest", "kstwo", "kstwobign", "kurtosistest", "landau",
        "laplace", "laplace_asymmetric", "levene", "levy", "levy_l", "levy_stable", "lmoment",
        "log", "loggamma", "logistic", "loglaplace", "lognorm", "logrank", "logser", "loguniform",
        "lomax", "make_distribution", "mannwhitneyu", "matrix_normal", "matrix_t", "maxwell",
        "median_abs_deviation", "median_test", "mielke", "monte_carlo_test", "mood", "morestats",
        "moyal", "mstats", "mstats_basic", "mstats_extras", "multinomial", "multiscale_graphcorr",
        "multivariate_hypergeom", "multivariate_normal", "multivariate_t", "mvn", "mvsdist",
        "nakagami", "nbinom", "ncf", "nchypergeom_fisher", "nchypergeom_wallenius", "nct", "ncx2",
        "nhypergeom", "normal_inverse_gamma", "normaltest", "norminvgauss", "obrientransform",
        "order_statistic", "ortho_group", "page_trend_test", "pareto", "pearson3",
        "percentileofscore", "permutation_test", "planck", "pmean", "pointbiserialr",
        "poisson_binom", "poisson_means_test", "power", "powerlaw", "powerlognorm", "powernorm",
        "ppcc_max", "ppcc_plot", "probplot", "qmc", "quantile", "quantile_test", "randint",
        "random_correlation", "random_table", "ranksums", "rayleigh", "rdist", "recipinvgauss",
        "reciprocal", "rel_breitwigner", "relfreq", "rice", "rv_histogram", "scoreatpercentile",
        "semicircular", "shapiro", "siegelslopes", "sigmaclip", "skellam", "skewcauchy",
        "skewnorm", "skewtest", "sobol_indices", "somersd", "spearmanrho", "special_ortho_group",
        "stats", "studentized_range", "theilslopes", "tiecorrect", "tmax", "tmean", "tmin",
        "trapezoid", "triang", "trim1", "trimboth", "truncate", "truncexpon", "truncnorm",
        "truncpareto", "truncweibull_min", "tsem", "tstd", "tukey_hsd", "tukeylambda", "tvar",
        "uniform_direction", "unitary_group", "variation", "vonmises", "vonmises_fisher",
        "vonmises_line", "wald", "wasserstein_distance", "wasserstein_distance_nd",
        "weibull_max", "weibull_min", "weightedtau", "wilcoxon", "wishart", "wrapcauchy",
        "yeojohnson", "yeojohnson_llf", "yeojohnson_normmax", "yeojohnson_normplot", "yulesimon",
        "zipf", "zipfian",
    }
)


def __getattr__(name):
    if name in _UNSUPPORTED_NAMES:
        raise NotImplementedError(f"scipy.stats.{name} is not supported by shellsim's SciPy")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
