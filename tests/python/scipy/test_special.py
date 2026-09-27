# Portable SciPy semantics. Expectations checked against SciPy 1.18.1 and NumPy 2.5.3 on
# CPython 3.14.4.
# Scope: scipy.special ufuncs (values, dtypes, edge cases) and its Python-level helpers.
# Values agree with SciPy to 1e-12 relative; SciPy computes some of them with Boost, so the last
# bits may differ.

import warnings

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from scipy import special


def close(actual, expected, rtol=1e-12):
    assert_allclose(actual, expected, rtol=rtol, atol=0)


def test_special_functions_are_numpy_ufuncs():
    assert isinstance(special.erf, np.ufunc)
    assert repr(special.erf) == "<ufunc 'erf'>"
    assert special.gammaln.__name__ == "gammaln"
    assert (special.erf.nin, special.beta.nin, special.betainc.nin) == (1, 2, 3)
    assert special.erf.nout == 1
    assert special.psi is special.digamma
    assert special.psi.__name__ == "psi"


def test_scalar_inputs_return_numpy_scalars():
    result = special.erf(0.5)
    assert type(result) is np.float64
    assert type(special.gamma(5)) is np.float64
    assert special.gamma(5) == 24.0
    assert type(special.erf(np.float32(0.5))) is np.float32


def test_loop_selection_follows_the_float_and_double_loops():
    assert special.erf(np.int8(1)).dtype == np.float32
    assert special.erf(np.float16(1)).dtype == np.float32
    assert special.erf(np.int32(1)).dtype == np.float64
    assert special.erf(np.array([True])).dtype == np.float32
    assert special.betainc(np.float32(2), 3, 0.5).dtype == np.float32
    assert special.gammainc(np.array([1, 2]), np.array([0.5, 1.5])).dtype == np.float64
    close(special.erf(np.float32(0.5)), 0.5204998778130465, rtol=1e-6)


def test_ufuncs_broadcast_and_write_to_out():
    a = np.array([[1.0], [2.0], [3.0]])
    x = np.array([0.5, 1.0])
    result = special.gammainc(a, x)
    assert result.shape == (3, 2)
    close(result[2], special.gammainc(3.0, x))
    out = np.empty(2)
    returned = special.erf(x, out=out)
    assert returned is out
    close(out, [0.5204998778130465, 0.8427007929497148])


def test_non_numeric_inputs_have_no_loop():
    message = "ufunc 'erf' not supported for the input types"
    with pytest.raises(TypeError, match=message):
        special.erf(np.array(["a"]))
    with pytest.raises(TypeError, match=message):
        special.erf(np.array([1], dtype=object))
    with pytest.raises(TypeError, match="ufunc 'erfinv' not supported"):
        special.erfinv(1j)


def test_poles_and_domain_errors_are_silent():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with np.errstate(all="raise"):
            assert special.gammaln(0) == np.inf
            assert special.logit(0) == -np.inf
            assert special.erfinv(1) == np.inf
            assert np.isnan(special.gamma(-1))
            assert np.isnan(special.betainc(1, 1, 2))


def test_erf_and_erfc():
    close(
        special.erf([-3.0, -1.0, -0.5, 0.0, 1e-10, 0.5, 1.0, 2.5, 6.0]),
        [
            -0.9999779095030014,
            -0.8427007929497148,
            -0.5204998778130465,
            0.0,
            1.1283791670955126e-10,
            0.5204998778130465,
            0.8427007929497148,
            0.999593047982555,
            1.0,
        ],
    )
    close(
        special.erfc([-2.0, 0.0, 0.5, 1.0, 5.0, 10.0, 26.0, 27.3]),
        [
            1.9953222650189528,
            1.0,
            0.4795001221869535,
            0.15729920705028516,
            1.5374597944280347e-12,
            2.0884875837625446e-45,
            5.663192408856145e-296,
            0.0,
        ],
    )


def test_erf_inverses():
    close(
        special.erfinv([-0.9, -0.5, 0.0, 1e-8, 0.5, 0.9, 0.999999]),
        [
            -1.1630871536766743,
            -0.4769362762044699,
            0.0,
            8.86226925452758e-09,
            0.4769362762044699,
            1.1630871536766743,
            3.458910737275499,
        ],
    )
    assert_array_equal(special.erfinv([-1.0, 1.0, 1.5, -2.0]), [-np.inf, np.inf, np.nan, np.nan])
    close(
        special.erfcinv([1e-300, 1e-10, 0.5, 1.0, 1.5, 1.999, 0.0, 2.0, 2.5]),
        [
            26.209469960516124,
            4.572824967389486,
            0.4769362762044699,
            -0.0,
            -0.4769362762044699,
            -2.3267537655135464,
            np.inf,
            -np.inf,
            np.nan,
        ],
    )
    x = np.linspace(-0.99, 0.99, 7)
    close(special.erf(special.erfinv(x)), x)


def test_gamma():
    close(
        special.gamma([0.5, 1.0, 2.5, 5.0, 10.1, 171.5, -0.5, -2.5, 1e-8]),
        [
            1.7724538509055159,
            1.0,
            1.329340388179137,
            24.0,
            454760.7514415855,
            9.483367566824803e307,
            -3.5449077018110318,
            -0.9453087204829419,
            99999999.42278434,
        ],
    )
    assert_array_equal(
        special.gamma([0.0, -0.0, -1.0, -2.0, 172.0, 171.7]),
        [np.inf, -np.inf, np.nan, np.nan, np.inf, np.inf],
    )
    close(
        special.rgamma([0.0, -1.0, 0.5, 3.0, -2.5, 180.0]),
        [0.0, 0.0, 0.5641895835477563, 0.5, -1.057855469152043, 0.0],
    )


def test_log_gamma():
    close(
        special.gammaln([0.5, 1.0, 2.0, 3.7, 100.0, 1e5, 1e300, -0.5, -1.5, 1e-300]),
        [
            0.5723649429247,
            0.0,
            0.0,
            1.428072326665388,
            359.1342053695754,
            1051287.7089736569,
            6.897755278982137e302,
            1.2655121234846454,
            0.860047015376481,
            690.7755278982137,
        ],
    )
    assert_array_equal(special.gammaln([0.0, -1.0, -2.0]), [np.inf, np.inf, np.inf])
    close(
        special.loggamma([0.5, 3.7, 100.0, 0.0]),
        [0.5723649429247, 1.428072326665388, 359.1342053695754, np.inf],
    )
    assert np.isnan(special.loggamma(-0.5))


def test_digamma():
    close(
        special.psi([1.0, 0.5, 2.5, 10.0, -0.5, 1e-8, 1e8, 0.0, -1.0]),
        [
            -0.5772156649015329,
            -1.9635100260214235,
            0.7031566406452432,
            2.251752589066721,
            0.03648997397857651,
            -100000000.57721564,
            18.420680738952367,
            -np.inf,
            np.nan,
        ],
    )


def test_beta_functions():
    close(
        special.beta([2.0, 0.5, 1e-3, 100.0], [3.0, 0.5, 2.0, 200.0]),
        [0.08333333333333333, 3.1415926535897927, 999.0009990009993, 3.607285449794492e-84],
    )
    close(
        special.betaln([2.0, 0.5, 100.0, 1e5], [3.0, 0.5, 200.0, 3.0]),
        [-2.4849066497880004, 1.1447298858494, -192.134192274979, -33.84565921407193],
    )


def test_regularized_incomplete_beta():
    a = [0.5, 2.0, 2.0, 10.0, 100.0, 1.0, 0.1]
    b = [0.5, 3.0, 3.0, 5.0, 150.0, 1.0, 0.2]
    x = [0.3, 0.0, 0.4, 0.7, 0.4, 0.25, 0.999]
    close(
        special.betainc(a, b, x),
        [
            0.36901011956554536,
            0.0,
            0.5247999999999999,
            0.5842011862193499,
            0.50343561029854,
            0.25,
            0.9139598779614837,
        ],
    )
    close(
        special.betaincc([0.5, 2.0, 10.0, 100.0], [0.5, 3.0, 5.0, 150.0], [0.3, 0.4, 0.7, 0.4]),
        [0.6309898804344546, 0.47519999999999996, 0.41579881378065014, 0.49656438970145966],
    )
    close(
        special.betaincinv([0.5, 2.0, 10.0, 100.0, 2.0], [0.5, 3.0, 5.0, 150.0, 3.0], [0.3, 0.5, 0.01, 0.99, 1.0]),
        [0.2061073738537634, 0.3857275681323895, 0.37256530511457264, 0.4729338136867745, 1.0],
    )


def test_regularized_incomplete_gamma():
    close(
        special.gammainc([0.5, 1.0, 2.0, 10.0, 100.0, 1e-3, 5.0], [0.3, 1.0, 5.0, 8.0, 110.0, 0.5, 0.0]),
        [
            0.5614219739190003,
            0.6321205588285577,
            0.9595723180054873,
            0.28337574127298903,
            0.8417213299399129,
            0.9994399333435291,
            0.0,
        ],
    )
    close(
        special.gammaincc([0.5, 1.0, 2.0, 10.0, 100.0, 20.0], [0.3, 1.0, 5.0, 8.0, 110.0, 60.0]),
        [
            0.4385780260809997,
            0.36787944117144245,
            0.04042768199451279,
            0.716624258727011,
            0.15827867006008706,
            6.35191834037896e-10,
        ],
    )
    close(
        special.gammaincinv([0.5, 1.0, 2.0, 10.0, 100.0], [0.3, 0.5, 0.9, 1e-5, 0.5]),
        [0.07423593091627269, 0.6931471805599455, 3.889720169867429, 1.6643376631140123, 99.66686491931549],
    )
    close(
        special.gammainccinv([0.5, 1.0, 2.0, 10.0, 100.0], [0.3, 0.5, 0.9, 1e-5, 0.5]),
        [0.5370970854287861, 0.6931471805599455, 0.5318116083896118, 29.522275193400823, 99.66686491931549],
    )
    assert np.isnan(special.gammainc(-1, 1))


def test_normal_distribution_functions():
    close(
        special.ndtr([-40.0, -10.0, -3.0, 0.0, 1.0, 1.959963984540054, 8.5]),
        [0.0, 7.61985302416047e-24, 0.0013498980316300933, 0.5, 0.8413447460685429, 0.975, 1.0],
    )
    close(
        special.log_ndtr([-40.0, -5.0, -1.0, 0.0, 5.0, 40.0]),
        [
            -804.6084420137539,
            -15.064998393988727,
            -1.8410216450092634,
            -0.6931471805599453,
            -2.8665161296376294e-07,
            -0.0,
        ],
    )
    close(
        special.ndtri([1e-300, 1e-10, 0.025, 0.5, 0.975, 1 - 1e-10, 0.0, 1.0, 1.5]),
        [
            -37.0470962993612,
            -6.361340902404056,
            -1.9599639845400545,
            0.0,
            1.959963984540054,
            6.361340889697422,
            -np.inf,
            np.inf,
            np.nan,
        ],
    )


def test_logistic_functions():
    close(
        special.expit([-800.0, -1.0, 0.0, 1.0, 800.0]),
        [0.0, 0.2689414213699951, 0.5, 0.7310585786300049, 1.0],
    )
    close(
        special.logit([0.0, 0.25, 0.5, 1.0, 1e-10, 2.0]),
        [-np.inf, -1.0986122886681098, 0.0, np.inf, -23.025850929840455, np.nan],
    )
    close(
        special.log_expit([-800.0, -1.0, 0.0, 1.0, 50.0]),
        [-800.0, -1.3132616875182228, -0.6931471805599453, -0.31326168751822286, -1.9287498479639178e-22],
    )


def test_information_theory_functions():
    close(special.xlogy([0.0, 2.0, 0.0, 3.0], [0.0, 3.0, np.nan, 0.0]), [0.0, 2.1972245773362196, np.nan, -np.inf])
    close(special.xlog1py([0.0, 2.0, 1.5], [-1.0, 1e-10, 3.0]), [0.0, 1.9999999999000001e-10, 2.0794415416798357])
    close(
        special.entr([-1.0, 0.0, 0.5, 1.0, 2.0]),
        [-np.inf, 0.0, 0.34657359027997264, -0.0, -1.3862943611198906],
    )
    close(
        special.rel_entr([0.5, 0.0, 1.0, 0.5], [0.25, 0.3, 0.0, -1.0]),
        [0.34657359027997264, 0.0, np.inf, np.inf],
    )
    close(special.kl_div([0.5, 0.0, 1.0], [0.25, 0.3, 0.0]), [0.09657359027997264, 0.3, np.inf])


def test_binom_and_poch():
    close(
        special.binom([10.0, 5.5, -3.0, 50.0, 3.0, 1e10], [3.0, 2.0, 2.0, 25.0, 5.0, 2.0]),
        [120.0, 12.375, np.nan, 126410606437752.03, 0.0, 4.9999999995e19],
    )
    close(
        special.poch([2.0, 0.5, 10.0, -2.5, 3.0], [3.0, 0.5, -2.0, 2.0, 0.0]),
        [24.0, 0.5641895835477564, 0.013888888888888888, 3.75, 1.0],
    )


def test_student_t_distribution_functions():
    close(
        special.stdtr([1.0, 3.0, 10.0, 2.5, 30.0], [0.5, -2.0, 1.812, 0.0, 3.0]),
        [0.6475836176504333, 0.06966298427942152, 0.9499623689670764, 0.5, 0.9973050179671741],
    )
    close(
        special.stdtrit([1.0, 3.0, 10.0, 2.5, 30.0], [0.75, 0.025, 0.95, 0.5, 0.999]),
        [1.0000000000000002, -3.1824463052837086, 1.8124611228116756, 0.0, 3.3851848668293045],
    )


def test_chi_square_distribution_functions():
    close(
        special.chdtr([1.0, 2.0, 10.0, 0.5, 100.0], [0.5, 3.0, 18.307, 0.01, 120.0]),
        [0.5204998778130466, 0.7768698398515702, 0.9499994109086018, 0.2930808947210196, 0.9155933189063082],
    )
    close(
        special.chdtrc([1.0, 2.0, 10.0, 100.0], [0.5, 3.0, 18.307, 120.0]),
        [0.47950012218695337, 0.22313016014842982, 0.05000058909139812, 0.08440668109369177],
    )
    close(
        special.chdtri([1.0, 2.0, 10.0, 100.0], [0.5, 0.05, 0.05, 0.9]),
        [0.4549364231195724, 5.991464547107983, 18.30703805327515, 82.35813581235715],
    )


def test_f_distribution_functions():
    dfn = [1.0, 5.0, 10.0, 2.0]
    dfd = [1.0, 2.0, 20.0, 30.0]
    close(
        special.fdtr(dfn, dfd, [1.0, 3.0, 2.35, 0.5]),
        [0.5000000000000001, 0.7313172949523805, 0.9501759161270916, 0.3885042917915453],
    )
    close(
        special.fdtrc(dfn, dfd, [1.0, 3.0, 2.35, 0.5]),
        [0.5000000000000001, 0.26868270504761954, 0.049824083872908424, 0.6114957082084547],
    )
    close(
        special.fdtri(dfn, dfd, [0.5, 0.95, 0.05, 0.99]),
        [1.0, 19.296409652017235, 0.3604881357605583, 5.390345863177884],
    )


def test_poisson_and_binomial_distribution_functions():
    close(
        special.pdtr([0.0, 2.0, 5.0, 10.0], [1.0, 3.0, 5.5, 20.0]),
        [0.36787944117144245, 0.42319008112684364, 0.5289186865258626, 0.010811718826652723],
    )
    close(
        special.pdtrc([0.0, 2.0, 5.0, 10.0], [1.0, 3.0, 5.5, 20.0]),
        [0.6321205588285577, 0.5768099188731566, 0.47108131347413745, 0.9891882811733472],
    )
    close(
        special.bdtr([0.0, 3.0, 7.0, 50.0], [10, 10, 20, 100], [0.3, 0.5, 0.25, 0.5]),
        [0.028247524899999984, 0.17187499999999997, 0.8981881430772772, 0.5397946186935897],
    )
    close(
        special.bdtrc([0.0, 3.0, 7.0, 50.0], [10, 10, 20, 100], [0.3, 0.5, 0.25, 0.5]),
        [0.9717524751000001, 0.828125, 0.10181185692272272, 0.4602053813064103],
    )


def binomial_warnings(function, *args):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        function(*args)
    return {(warning.category, str(warning.message)) for warning in caught}


def test_binomial_distribution_deprecates_non_integer_trials():
    # SciPy warns once per element and shellsim once per call, so the test compares sets.
    deprecated = {(DeprecationWarning, "non-integer arg n is deprecated, removed in SciPy 1.7.x")}
    assert binomial_warnings(special.bdtr, 3.0, np.array([10.0, 20.0]), 0.5) == deprecated
    # Single-precision operands select the float32 loop, which takes n as a float.
    trials = np.array([10, 20], dtype=np.int16)
    assert binomial_warnings(special.bdtrc, np.float32(3.0), trials, np.float32(0.5)) == deprecated
    assert binomial_warnings(special.bdtr, 3.0, np.array([10, 20]), 0.5) == set()
    assert binomial_warnings(special.bdtrc, 3.0, 10, 0.5) == set()
    assert binomial_warnings(special.bdtr, 3.0, np.array([], dtype=float), 0.5) == set()


def test_box_cox_transform():
    close(special.boxcox([1.0, 2.0, 10.0, 0.5], [0.0, 0.5, -1.0, 2.0]), [0.0, 0.8284271247461901, 0.9, -0.375])
    close(
        special.inv_boxcox([0.0, 0.8284271247461903, 0.9, -0.375], [0.0, 0.5, -1.0, 2.0]),
        [1.0, 2.0, 10.000000000000002, 0.5],
    )


def test_zeta():
    close(
        special.zeta([2.0, 3.0, 1.5, 10.0, 1.0, 0.5, -1.0, -2.0]),
        [
            1.6449340668482264,
            1.2020569031595942,
            2.612375348685488,
            1.000994575127818,
            np.inf,
            -1.4603545088095866,
            -0.08333333333333338,
            0.0,
        ],
    )
    close(
        special.zeta([2.0, 3.0, 1.5], [2.0, 0.5, 10.0]),
        [0.6449340668482266, 8.414398322117158, 0.6486616319415703],
    )


def test_softmax_and_log_softmax():
    close(special.softmax([1.0, 2.0, 3.0]), [0.09003057317038046, 0.24472847105479764, 0.6652409557748218])
    m = [[1.0, 2.0], [3.0, 5.0]]
    close(
        special.softmax(m, axis=0),
        [[0.11920292202211755, 0.04742587317756679], [0.8807970779778823, 0.9525741268224334]],
    )
    close(
        special.softmax(m, axis=1),
        [[0.2689414213699951, 0.7310585786300049], [0.11920292202211755, 0.8807970779778823]],
    )
    close(
        special.softmax(m),
        [[0.015219428864155926, 0.04137069692096015], [0.11245721367093255, 0.8309526605439513]],
    )
    close(special.softmax([1000.0, 1001.0]), [0.2689414213699951, 0.7310585786300049])
    close(
        special.log_softmax([1.0, 2.0, 3.0]),
        [-2.4076059644443806, -1.4076059644443804, -0.4076059644443804],
    )
    close(
        special.log_softmax(m, axis=1),
        [[-1.3132616875182228, -0.31326168751822286], [-2.1269280110429727, -0.1269280110429726]],
    )


def test_logsumexp():
    close(special.logsumexp([1.0, 2.0, 3.0]), 3.40760596444438)
    m = [[1.0, 2.0], [3.0, 5.0]]
    close(special.logsumexp(m, axis=0), [3.1269280110429727, 5.048587351573742])
    kept = special.logsumexp(m, axis=1, keepdims=True)
    assert kept.shape == (2, 1)
    close(kept, [[2.313261687518223], [5.126928011042972]])
    close(special.logsumexp([1000.0, 1000.0]), 1000.6931471805599)
    assert special.logsumexp([-np.inf, -np.inf]) == -np.inf
    assert special.logsumexp([np.inf, 1.0]) == np.inf
    close(special.logsumexp([1.0, 2.0], b=[0.5, 2.0]), 2.7811304570373454)
    value, sign = special.logsumexp([1.0, 2.0], b=[1.0, -1.0], return_sign=True)
    close(value, 1.5413248546129181)
    assert sign == -1.0
    value, sign = special.logsumexp([1.0, 1.0], b=[1.0, -1.0], return_sign=True)
    assert (value, sign) == (-np.inf, 0.0)


def test_comb_and_perm():
    assert special.comb(5, 2) == 10.0
    assert type(special.comb(5, 2)) is np.float64
    result = special.comb(5, 2, exact=True)
    assert result == 10 and type(result) is int
    assert_array_equal(special.comb([10, 10, 10, 4], [0, 3, 11, -1]), [1.0, 120.0, 0.0, 0.0])
    assert special.comb(30, 15, exact=True) == 155117520
    close(special.comb(100, 50), 1.0089134454556415e29)
    assert special.comb(100, 50, exact=True) == 100891344545564193334812497256
    assert special.comb(5, 3, exact=True, repetition=True) == 35
    assert_array_equal(special.comb(np.array([5, 6]), 2, repetition=True), [15.0, 21.0])
    assert special.comb(-1, 2) == 0.0
    assert_array_equal(special.perm([5, 5, 5], [0, 2, 6]), [1.0, 20.0, 0.0])
    assert special.perm(10, 3, exact=True) == 720
    with pytest.raises(ValueError, match="Non-integer `N` and `k` with `exact=True` is not supported."):
        special.comb(5.5, 2, exact=True)


def test_factorial():
    result = special.factorial(5)
    assert result == 120.0 and type(result) is np.float64
    assert special.factorial(5, exact=True) == 120
    close(
        special.factorial([0, 1, 5, 10, 20, -3]),
        [1.0, 1.0, 120.0, 3628800.0, 2.43290200817664e18, 0.0],
    )
    assert special.factorial(25, exact=True) == 15511210043330985984000000
    assert_array_equal(special.factorial(np.array([[1, 2], [3, 4]])), [[1.0, 2.0], [6.0, 24.0]])
    exact = special.factorial(np.array([[1, 2], [3, 4]]), exact=True)
    assert exact.dtype == np.int64
    assert_array_equal(exact, [[1, 2], [6, 24]])
    close(special.factorial(2.5), 3.323350970447843)
    close(special.factorial([1.5, 2.0]), [1.329340388179137, 2.0])
    assert special.factorial(171) == np.inf
    with pytest.raises(ValueError, match="`exact=True` only supports integers"):
        special.factorial(0.5, exact=True)
