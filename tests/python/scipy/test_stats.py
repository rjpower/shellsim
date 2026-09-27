# Portable SciPy semantics. Expectations checked against SciPy 1.18.1 and NumPy 2.5.3 on
# CPython 3.14.4.
# Scope: scipy.stats distributions, summary statistics, correlation and hypothesis tests.
# Values agree with SciPy to 1e-12 relative; SciPy computes some of them with Boost, so the last
# bits may differ.

import warnings

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from scipy import stats


def close(actual, expected, rtol=1e-12):
    assert_allclose(actual, expected, rtol=rtol, atol=0)


X = np.array([[1.0, 2.0, 3.0, 10.0], [2.0, 2.5, 0.5, 4.0], [3.0, 1.0, 1.0, 7.0]])

# name: (shape arguments, pdf at [0.2, 1.0, 2.5] with loc=0.1 and scale=1.5, cdf, ppf at
# [0.01, 0.5, 0.975], stats(moments="mvsk"), entropy, rvs(size=3, random_state=1)).
CONTINUOUS = {
    "norm": (
        (),
        [0.26537115087596813, 0.22214973526119977, 0.07394722311963707],
        [0.579259709439103, 0.8413447460685429, 0.9937903346742238],
        [-2.3263478740408408, 0.0, 1.959963984540054],
        (0.0, 1.0, 0.0, 0.0),
        1.4189385332046727,
        [1.6243453636632417, -0.6117564136500754, -0.5281717522634557],
    ),
    "t": (
        (5,),
        [0.25239746818043485, 0.2054273398156132, 0.073212835103994],
        [0.5753197430020855, 0.8183912661754386, 0.9727549503288119],
        [-3.364929998907218, 0.0, 2.5705818356363146],
        (0.0, 1.6666666666666667, 0.0, 6.0),
        1.627502672414396,
        [2.18220870791557, -1.0119851341385129, -0.8613659557539434],
    ),
    "chi2": (
        (3,),
        [0.06641966709291876, 0.1526181157536372, 0.15116220299330552],
        [0.02241070223835061, 0.19874804309879915, 0.5247089166569795],
        [0.11483180189911707, 2.3659738843753377, 9.348403604496148],
        (3.0, 6.0, 1.632993161855452, 4.0),
        2.0541199559354117,
        [7.8952381420326745, 1.2455846702395579, 1.3682242944803054],
    ),
    "f": (
        (4, 7),
        [0.18608756319076092, 0.4065373738269238, 0.15423964373681784],
        [0.0694851825241943, 0.5327857658636862, 0.8629666302475232],
        [0.06677458461437806, 0.9261930995100327, 5.5225943453085495],
        (1.4, 2.94, 10.61445555206044, np.nan),
        1.27679852186135,
        [3.7946615973238567, 1.1630501858707263, 4.459387701825801],
    ),
    "uniform": (
        (),
        [0.6666666666666666, 0.6666666666666666, 0.0],
        [0.2, 1.0, 1.0],
        [0.01, 0.5, 0.975],
        (0.5, 0.08333333333333333, 0.0, -1.2),
        0.0,
        [0.417022004702574, 0.7203244934421581, 0.00011437481734488664],
    ),
    "expon": (
        (),
        [0.6236713233544119, 0.3658744240626843, 0.13459767866310363],
        [0.18126924692201815, 0.6321205588285577, 0.9179150013761012],
        [0.010050335853501442, 0.6931471805599453, 3.6888794541139354],
        (1.0, 1.0, 2.0, 6.0),
        1.0,
        [0.5396058372591854, 1.2741252530133043, 0.00011438135864308592],
    ),
}


@pytest.mark.parametrize("name", ["chi2", "expon", "f", "norm", "t", "uniform"])
def test_continuous_distribution_values(name):
    dist = getattr(stats, name)
    args, pdf, cdf, ppf, moments, entropy, rvs = CONTINUOUS[name]
    x = [0.2, 1.0, 2.5]
    q = [0.01, 0.5, 0.975]
    close(dist.pdf(x, *args, loc=0.1, scale=1.5), pdf)
    close(dist.cdf(x, *args), cdf)
    close(dist.sf(x, *args), 1 - np.array(cdf), rtol=1e-9)
    close(dist.ppf(q, *args), ppf)
    close(dist.isf(1 - np.array(q), *args), ppf, rtol=1e-9)
    close(dist.logpdf(0.7, *args), np.log(dist.pdf(0.7, *args)))
    close(np.array(dist.stats(*args, moments="mvsk"), dtype=float), moments)
    close(dist.entropy(*args), entropy)
    close(dist.rvs(*args, size=3, random_state=1), rvs)


def test_continuous_results_are_numpy_scalars_or_arrays():
    assert type(stats.norm.cdf(1.0)) is np.float64
    assert stats.norm.cdf([1.0]).dtype == np.float64
    assert stats.norm.cdf([[0.0, 1.0]]).shape == (1, 2)
    mean, var = stats.t.stats(5)
    assert (type(mean), type(var)) == (np.float64, np.float64)


def test_distribution_summaries_and_frozen_distributions():
    close(stats.norm.interval(0.9), (-1.6448536269514729, 1.6448536269514722))
    close(stats.chi2.median(3), 2.3659738843753377)
    assert stats.norm.mean(loc=2) == 2.0
    assert stats.t.var(5, scale=3) == 15.0
    assert stats.expon.moment(2) == 2.0
    assert stats.chi2.support(3) == (0.0, np.inf)
    frozen = stats.norm(loc=1, scale=2)
    close(frozen.cdf(1.5), 0.5987063256829237)
    assert frozen.mean() == 1.0
    assert frozen.std() == 2.0
    close(stats.t(5).ppf(0.975), 2.5705818356363146)


def test_invalid_parameters_give_nan_and_logpdf_outside_support_is_minus_infinity():
    assert np.isnan(stats.norm.cdf(0.5, scale=-1))
    assert np.isnan(stats.t.pdf(0.5, -2))
    assert np.isnan(stats.norm.cdf(np.nan))
    assert stats.uniform.logpdf(1.3) == -np.inf
    assert stats.expon.pdf(-1.0) == 0.0


def test_argument_binding_matches_scipy_errors():
    with pytest.raises(TypeError, match="missing 1 required positional argument: 'df'"):
        stats.t.pdf(0.5)
    with pytest.raises(TypeError, match="got multiple values for argument 'loc'"):
        stats.norm.pdf(0.5, 1, loc=1)
    with pytest.raises(TypeError, match="got an unexpected keyword argument 'dfx'"):
        stats.t.cdf(0.5, dfx=3)


# name: (shape arguments, pmf at [0, 3, 7], cdf, ppf at [0.01, 0.5, 0.99], stats("mvsk"),
# entropy, expect of k**2, rvs(size=4, random_state=default_rng(3))).
DISCRETE = {
    "binom": (
        (10, 0.3),
        [0.0282475249, 0.2668279319999998, 0.009001691999999992],
        [0.028247524900000005, 0.6496107184000002, 0.9984096136],
        [0.0, 3.0, 7.0],
        (3.0, 2.1, 0.27602622373694163, -0.12380952380952381),
        1.7790787840900626,
        11.099999999999994,
        [1, 2, 4, 3],
    ),
    "poisson": (
        (2.5,),
        [0.0820849986238988, 0.21376301724973648, 0.009940616501568845],
        [0.0820849986238988, 0.7575761331330662, 0.9957533045106555],
        [0.0, 2.0, 7.0],
        (2.5, 2.5, 0.6324555320336759, 0.4),
        1.8307266079269722,
        8.750000000000002,
        [1, 2, 2, 2],
    ),
}


@pytest.mark.parametrize("name", ["binom", "poisson"])
def test_discrete_distribution_values(name):
    dist = getattr(stats, name)
    args, pmf, cdf, ppf, moments, entropy, expect, rvs = DISCRETE[name]
    close(dist.pmf([0, 3, 7], *args), pmf)
    close(dist.cdf([0, 3, 7], *args), cdf)
    close(dist.sf([0, 3, 7], *args), 1 - np.array(cdf), rtol=1e-9)
    assert dist.ppf([0.01, 0.5, 0.99], *args).tolist() == ppf
    close(np.array(dist.stats(*args, moments="mvsk"), dtype=float), moments)
    close(dist.entropy(*args), entropy)
    close(dist.expect(lambda k: k**2, args), expect)
    assert dist.rvs(*args, size=4, random_state=np.random.default_rng(3)).tolist() == rvs


def test_closed_form_fits():
    data = [1.0, 2.0, 4.0, 7.0]
    close(stats.norm.fit(data), (3.5, 2.29128784747792))
    close(stats.norm.fit(data, floc=0), (0.0, 4.183300132670378))
    assert stats.expon.fit(data) == (1.0, 2.5)
    assert stats.uniform.fit(data) == (1.0, 6.0)
    with pytest.raises(TypeError, match="takes 2 positional arguments but 3 were given"):
        stats.norm.fit(data, 3)


class _ShapedGen(stats.rv_continuous):
    def _pdf(self, x, k):
        return k * np.exp(-k * x)


class _FromCdfGen(stats.rv_continuous):
    def _cdf(self, x):
        return -np.expm1(-x)


def test_subclasses_take_shapes_from_their_method_signatures():
    shaped = _ShapedGen(a=0.0, name="shaped")
    assert (shaped.numargs, shaped.shapes) == (1, "k")
    close(shaped.pdf(1.0, 2.0), 0.2706705664732254)
    explicit = _ShapedGen(a=0.0, name="explicit", shapes="rate")
    assert explicit.shapes == "rate"
    close(explicit.pdf(1.0, rate=2.0), 0.2706705664732254)
    close(_FromCdfGen(a=0.0, name="from_cdf").pdf(1.0), 0.36787944117100396, rtol=1e-9)


class _DefaultShapeGen(stats.rv_continuous):
    def _pdf(self, x, k=1.0):
        return k * np.exp(-k * x)


class _InconsistentGen(stats.rv_continuous):
    def _pdf(self, x, k):
        return k * np.exp(-k * x)

    def _cdf(self, x, rate):
        return -np.expm1(-rate * x)


class _VarargsShapeGen(stats.rv_continuous):
    def _pdf(self, x, k, *extra):
        return k * np.exp(-k * x)


_BAD_SHAPES = {
    "defaults": _DefaultShapeGen,
    "inconsistent": _InconsistentGen,
    "varargs": _VarargsShapeGen,
}


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("defaults", "defaults are not allowed for shapes"),
        ("inconsistent", "Shape arguments are inconsistent."),
        ("varargs", r"\*args are not allowed w/out explicit shapes"),
    ],
)
def test_shape_inference_rejects_what_scipy_rejects(case, message):
    with pytest.raises(TypeError, match=message):
        _BAD_SHAPES[case](name="bad")


def test_describe():
    result = stats.describe(X, axis=1)
    assert result.nobs == 4
    assert_array_equal(result.minmax[0], [1.0, 0.5, 1.0])
    assert_array_equal(result.minmax[1], [10.0, 4.0, 7.0])
    close(result.mean, [4.0, 2.25, 3.0])
    close(result.variance, [16.666666666666668, 2.0833333333333335, 8.0])
    close(result.skewness, [1.0182337649086284, 0.0, 0.816496580927726], rtol=1e-9)
    close(result.kurtosis, [-0.7696, -1.0784, -1.0], rtol=1e-9)
    nobs, minmax, mean, variance, skewness, kurtosis = stats.describe([1, 2, 3, 4.5])
    assert type(nobs) is np.int64
    assert (mean, variance) == (2.625, 2.2291666666666665)
    with pytest.raises(ValueError, match="The input must not be empty."):
        stats.describe([])


def test_result_objects_unpack_and_expose_fields():
    result = stats.describe([1, 2, 3, 4.5])
    assert len(result) == 6
    assert result[2] == result.mean
    assert repr(stats.mode([1, 2, 2, 3])) == "ModeResult(mode=np.int64(2), count=np.int64(2))"
    assert stats.mode([1, 2, 2, 3])._asdict() == {"mode": 2, "count": 2}
    regression = stats.linregress([1, 2, 3, 4], [2, 1, 4, 3])
    slope, intercept, rvalue, pvalue, stderr = regression
    assert (slope, intercept) == (regression.slope, regression.intercept)
    close(regression.intercept_stderr, 1.5491933384829668)


def test_moments_skew_and_kurtosis():
    close(stats.skew(X, axis=1, bias=False), [1.763632614803888, 0.0, 1.4142135623730951])
    close(stats.kurtosis(X, axis=1, fisher=False, bias=False), [6.228, 3.912, 4.5])
    close(stats.moment(X, order=[2, 3], axis=1), [[12.5, 1.5625, 6.0], [45.0, 0.0, 12.0]])
    assert_array_equal(stats.moment(X, order=1, axis=1), [0.0, 0.0, 0.0])
    assert stats.moment([1, 2, 3, 4], order=2, center=0) == 7.5
    assert stats.skew(X, keepdims=True).shape == (1, 4)
    close(stats.skew(X, axis=None), 1.4861362886809195)


def test_nan_policies():
    assert np.isnan(stats.skew([1, 2, np.nan, 9]))
    close(stats.skew([1, 2, np.nan, 9], nan_policy="omit"), 0.6654688661238353)
    with pytest.raises(ValueError, match="The input contains nan values"):
        stats.skew([1, np.nan], nan_policy="raise")
    with pytest.raises(ValueError, match="nan_policy must be one of"):
        stats.skew([1, 2], nan_policy="ignore")
    close(
        stats.zscore([1, 2, np.nan, 4], nan_policy="omit")[[0, 1, 3]],
        [-1.0690449676496978, -0.2672612419124245, 1.3363062095621219],
    )


def test_mode_sem_zscore_and_trim_mean():
    result = stats.mode([[1, 2, 2], [3, 3, 1]], axis=1)
    assert_array_equal(result.mode, [2, 3])
    assert_array_equal(result.count, [2, 2])
    assert stats.mode([1, np.nan, np.nan, 2]).count == 2
    close(stats.sem(X, axis=1, ddof=0), [1.7677669529663689, 0.625, 1.224744871391589])
    close(
        stats.zscore(X[2], ddof=1), [0.0, -0.7071067811865475, -0.7071067811865475, 1.414213562373095]
    )
    close(stats.zmap([1, 2, 3], [2, 4, 6, 8]), [-1.7888543819998317, -1.3416407864998738, -0.8944271909999159])
    assert stats.trim_mean([1, 2, 3, 4, 5, 100], 0.2) == 3.5


SMALL_SAMPLE = (
    "One or more sample arguments is too small; all returned values will be NaN. "
    "See documentation for sample size requirements."
)
PRECISION_LOSS = (
    "Precision loss occurred in moment calculation due to catastrophic cancellation. "
    "This occurs when the data are nearly identical. Results may be unreliable."
)


def recorded_warnings(function, *args):
    """Call `function` and return its result with each warning's category name and message."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = function(*args)
    return result, [(warning.category.__name__, str(warning.message)) for warning in caught]


def test_small_samples_warn_and_give_nan():
    result, caught = recorded_warnings(stats.sem, [1.0])
    assert np.isnan(result)
    assert caught == [("SmallSampleWarning", SMALL_SAMPLE)]
    empty, caught = recorded_warnings(stats.mode, [])
    assert caught == [("SmallSampleWarning", SMALL_SAMPLE)]
    assert np.isnan(empty.mode)
    assert empty.count == 0


def test_constant_input_warns_about_precision_and_gives_nan():
    result, caught = recorded_warnings(stats.skew, [2.0, 2.0, 2.0])
    assert np.isnan(result)
    assert caught == [("RuntimeWarning", PRECISION_LOSS)] * 2


@pytest.mark.parametrize(
    ("method", "expected"),
    [
        ("average", [3.0, 1.5, 4.0, 1.5, 5.0]),
        ("min", [3.0, 1.0, 4.0, 1.0, 5.0]),
        ("max", [3.0, 2.0, 4.0, 2.0, 5.0]),
        ("dense", [2.0, 1.0, 3.0, 1.0, 4.0]),
        ("ordinal", [3.0, 1.0, 4.0, 2.0, 5.0]),
    ],
)
def test_rankdata(method, expected):
    assert stats.rankdata([3, 1, 4, 1, 5], method=method).tolist() == expected


def test_rankdata_axis_and_nan():
    assert stats.rankdata([[3, 1, 4], [1, 5, 5]], axis=1).tolist() == [[2.0, 1.0, 3.0], [1.0, 2.5, 2.5]]
    ranks = stats.rankdata([3, np.nan, 1], nan_policy="omit")
    assert ranks[0] == 2.0 and np.isnan(ranks[1]) and ranks[2] == 1.0
    with pytest.raises(ValueError, match='unknown method "middle"'):
        stats.rankdata([1, 2], method="middle")


def test_entropy():
    close(stats.entropy([0.2, 0.8]), 0.5004024235381879)
    close(stats.entropy([1, 2, 3], base=2), 1.459147917027245)
    close(stats.entropy([0.2, 0.8], [0.5, 0.5]), 0.19274475702175747)
    close(stats.entropy(X, axis=1), [1.0408398374232388, 1.211044016780123, 1.0751393240053733])
    with pytest.raises(ValueError, match="`base` must be a positive number or `None`."):
        stats.entropy([0.5, 0.5], base=-1)


def test_pearsonr():
    result = stats.pearsonr([1, 2, 3, 4, 5], [2, 1, 4, 3, 7])
    close(result.statistic, 0.824163383692134)
    close(result.pvalue, 0.08613863131395952)
    assert result.correlation == result.statistic
    close(result.confidence_interval(), (-0.2129343655851338, 0.9880137261479831))
    less = stats.pearsonr([1, 2, 3, 4, 5], [2, 1, 4, 3, 7], alternative="less")
    close(less.confidence_interval(0.9), (-1.0, 0.9690126698884127))
    close(stats.pearsonr(X, X[::-1], axis=1).statistic, [0.8660254037844388, 1.0, 0.8660254037844388])
    with pytest.raises(ValueError, match="`x` and `y` must have length at least 2."):
        stats.pearsonr([1], [2])
    constant, caught = recorded_warnings(stats.pearsonr, [1, 1, 1], [1, 2, 3])
    assert np.isnan(constant.statistic)
    message = "An input array is constant; the correlation coefficient is not defined."
    assert caught == [("ConstantInputWarning", message)]


def test_spearmanr_and_linregress():
    result = stats.spearmanr([1, 2, 3, 4], [2, 1, 4, 3])
    close(result.statistic, 0.6)
    close(result.pvalue, 0.4)
    matrix = stats.spearmanr(X, axis=1)
    close(matrix.statistic[0], [1.0, 0.4, 0.316227766016838])
    close(matrix.pvalue[1], [0.6, 0.0, 0.367544467966324], rtol=1e-9)
    regression = stats.linregress(X[0], X[1])
    close(
        tuple(regression) + (regression.intercept_stderr,),
        (0.25, 1.25, 0.7071067811865475, 0.29289321881345254, 0.1767766952966369, 0.9437293044088437),
    )


def test_one_sample_and_paired_t_tests():
    result = stats.ttest_1samp([5.1, 4.9, 5.6, 5.8, 6.0], 5.0)
    close((result.statistic, result.pvalue), (2.304073731539131, 0.08256829674577398))
    assert result.df == 4
    close(result.confidence_interval(0.9), (5.035879465387927, 5.924120534612071))
    columns = stats.ttest_1samp(X, [1.0, 2.0, 3.0, 4.0])
    close(columns.statistic, [1.7320508075688772, -0.3779644730092272, -1.9639610121239315, 1.7320508075688772])
    assert_array_equal(columns.df, [2, 2, 2, 2])
    paired = stats.ttest_rel([1, 2, 3, 4], [1.5, 2.2, 3.9, 4.1], alternative="greater")
    close((paired.statistic, paired.pvalue), (-2.3650683683768574, 0.9505277015229879))


@pytest.mark.parametrize(
    ("kwargs", "samples", "expected"),
    [
        ({}, ([1, 2, 3, 4], [3, 4, 5, 7.5]), (-2.0448636095023995, 0.08685750213603272, 6.0)),
        (
            {"equal_var": False},
            ([1, 2, 3, 4], [3, 4, 5, 7.5]),
            (-2.0448636095023995, 0.09373537498459526, 5.235113550636039),
        ),
        (
            {"trim": 0.2},
            ([1, 2, 3, 4, 20], [3, 4, 5, 7.5, 9]),
            (-1.4985022462565507, 0.20836841891042446, 4.0),
        ),
    ],
)
def test_independent_t_tests(kwargs, samples, expected):
    result = stats.ttest_ind(*samples, **kwargs)
    close((result.statistic, result.pvalue, result.df), expected)


def test_t_test_from_statistics_and_confidence_interval():
    close(
        stats.ttest_ind([1, 2, 3, 4], [3, 4, 5, 7.5]).confidence_interval(),
        (-5.2169575855640975, 0.46695758556409706),
    )
    close(
        stats.ttest_ind_from_stats(1.0, 0.5, 10, 1.4, 0.7, 12),
        (-1.5114980543804881, 0.14630183499566013),
    )


def test_chisquare_and_power_divergence():
    close(stats.chisquare([16, 18, 16, 14, 12, 12]), (2.0, 0.8491450360846096))
    close(stats.chisquare([16, 18, 16, 14, 12, 12], [16, 16, 16, 16, 16, 8]), (3.5, 0.6233876277495822))
    close(
        stats.power_divergence([16, 18, 16, 14, 12, 12], lambda_="log-likelihood"),
        (2.006573162632538, 0.8482347677946377),
    )
    with pytest.raises(ValueError, match="the sum of the observed frequencies must agree"):
        stats.chisquare([16, 18], [10, 10])
    with pytest.raises(ValueError, match="invalid string for lambda_: 'nope'"):
        stats.power_divergence([16, 18], lambda_="nope")


def test_contingency_tables():
    result = stats.chi2_contingency([[10, 20, 30], [6, 9, 17], [3, 4, 9]])
    close((result.statistic, result.pvalue), (0.5501644736842111, 0.9684372774263162))
    assert result.dof == 4
    close(result.expected_freq[0], [10.555555555555555, 18.333333333333332, 31.11111111111111])
    statistic, pvalue, dof, expected = stats.chi2_contingency([[10, 20], [30, 25]])
    close((statistic, pvalue), (2.706155303030302, 0.0999616438735349))
    assert dof == 1
    close(
        stats.chi2_contingency([[10, 20], [30, 25]], correction=False)[:2],
        (3.505892255892255, 0.06115089757606777),
    )
    assert stats.chi2_contingency([3, 4, 5])[:3] == (0.0, 1.0, 0)
    with pytest.raises(ValueError, match="has a zero element at"):
        stats.chi2_contingency([[0, 0], [1, 2]])
    table = [[10, 20, 5], [30, 25, 7]]
    close(stats.contingency.association(table), 0.19416079083690585)
    close(stats.contingency.association(table, method="tschuprow"), 0.16326911299758037)
    close(stats.contingency.association(table, method="pearson"), 0.19060134284134891)
    margins = stats.contingency.margins(np.array([[10, 20], [30, 25]]))
    assert [margin.tolist() for margin in margins] == [[[30], [55]], [[40, 45]]]
