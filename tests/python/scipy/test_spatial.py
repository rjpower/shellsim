# Portable SciPy semantics. Expectations checked against SciPy 1.18.1 and NumPy 2.5.3 on
# CPython 3.14.4.
# Scope: scipy.spatial.distance vector metrics, cdist, pdist and the condensed-matrix helpers.
# cdist and pdist sum left to right as SciPy's compiled loops do, so the sums below agree bit
# for bit; cosine, correlation, seuclidean and mahalanobis can differ from SciPy's compiled
# loops in the last bit and are compared with a tolerance.

import warnings

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from scipy.spatial import distance

import scipy.spatial

U = [1.0, 2.0, 5.0]
V = [4.0, 0.0, -1.0]
# Thirds on a shared grid: the inputs repeat and include zeros, and their sums round.
XA = (np.arange(60) * 7 % 11 - 5).reshape(6, 10) / 3
XB = (np.arange(50) * 3 % 7 - 3).reshape(5, 10) / 3
VI = np.eye(10) + 1 / 7


@pytest.mark.parametrize(
    "metric, expected",
    [
        ("braycurtis", 1.0),
        ("canberra", 2.6),
        ("chebyshev", 6.0),
        ("cityblock", 11.0),
        ("correlation", 1.8170571691028834),
        ("cosine", 1.0442807442770048),
        ("euclidean", 7.0),
        ("sqeuclidean", 49.0),
        ("hamming", 1.0),
        ("jaccard", 1 / 3),
        ("minkowski", 7.0),
    ],
)
def test_vector_metrics(metric, expected):
    result = getattr(distance, metric)(U, V)
    assert result == expected and isinstance(result, np.float64)


def test_vector_metrics_keep_or_promote_the_input_dtype():
    ints = ([1, 2], [3, 5])
    assert repr(distance.cityblock(*ints)) == "np.int64(5)"
    assert repr(distance.chebyshev(*ints)) == "np.int64(3)"
    assert repr(distance.sqeuclidean(*ints)) == "np.float64(13.0)"
    floats = (np.float32([1, 2]), np.float32([2, 4]))
    assert repr(distance.euclidean(*floats)) == "np.float32(2.236068)"
    assert repr(distance.braycurtis(*floats)) == "np.float64(0.3333333333333333)"
    # Like SciPy's, the vector functions reduce the last axis of higher-dimensional input.
    assert_array_equal(distance.euclidean([[0, 0], [1, 1]], [[3, 4], [1, 1]]), [5.0, 0.0])


def test_weighted_metrics():
    u, v, w = [1.0, 2.0], [3.0, 5.0], [1.0, 2.0]
    assert distance.cityblock(u, v, w=w) == 8.0
    assert distance.sqeuclidean(u, v, w=w) == 22.0
    assert distance.euclidean(u, v, w=w) == np.sqrt(22.0)
    assert_allclose(distance.minkowski(u, v, 3, w=w), 62.0 ** (1 / 3), rtol=1e-15)
    assert distance.chebyshev([1, 5, 2], [3, 4, 9], w=[1, 1, 0]) == 2
    assert distance.hamming([1, 0, 1], [1, 1, 0], w=[1, 2, 3]) == 0.8333333333333333
    assert distance.jaccard([1, 0, 1], [1, 1, 0], w=[1, 2, 3]) == 0.8333333333333334
    assert_allclose(distance.cosine(u, v, w=w), 0.0018850158136837214, rtol=1e-12)
    with pytest.raises(ValueError, match="^Input weights should be all non-negative$"):
        distance.euclidean(u, v, w=[1.0, -1.0])


def test_metric_edge_cases():
    assert distance.minkowski([1, 2], [3, 5], np.inf) == 3.0
    assert distance.minkowski([1, 2], [3, 5], 1) == 5.0
    assert distance.minkowski([1, 2], [3, 5], 0.5) == 9.898979485566358
    with pytest.raises(ValueError, match="^p must be greater than 0$"):
        distance.minkowski(U, V, 0)
    assert distance.cosine([1e-9, 3], [1e-9, 3]) == 0.0
    assert distance.correlation([1, 2, 3], [3, 2, 1]) == 2.0
    assert distance.canberra([0, 1], [0, 2]) == 1 / 3
    assert distance.jaccard([0, 0], [0, 0]) == 0.0
    assert distance.jaccard([True, False], [False, False]) == 1.0
    assert distance.seuclidean([1, 2], [3, 4], [1, 2]) == np.sqrt(6.0)
    assert distance.mahalanobis([1, 2], [3, 4], np.eye(2) * 2) == 4.0
    with pytest.raises(TypeError, match="^V must be a 1-D array of the same dimension as u and v.$"):
        distance.seuclidean([1, 2], [3, 4], [1, 2, 3])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert np.isnan(distance.cosine([0, 0], [1, 1]))
        assert np.isnan(distance.braycurtis([0, 0], [0, 0]))
    assert [warning.category for warning in caught] == [RuntimeWarning, RuntimeWarning]


def test_jensenshannon():
    assert distance.jensenshannon([1, 2], [3, 4]) == 0.0694067875446836
    assert distance.jensenshannon([1, 0], [0, 1]) == 0.8325546111576977
    assert distance.jensenshannon([1, 0], [0, 1], base=2) == 1.0
    rows = distance.jensenshannon([[1, 0], [0.5, 0.5]], [[0, 1], [0.5, 0.5]], axis=1, keepdims=True)
    assert_allclose(rows, [[0.8325546111576977], [0.0]], rtol=1e-15)
    with pytest.raises(np.exceptions.AxisError, match="^axis 1 is out of bounds for array of dimension 1$"):
        distance.jensenshannon([1, 2], [3, 4], axis=1)


@pytest.mark.parametrize(
    "metric, kwargs",
    [
        ("euclidean", {}),
        ("sqeuclidean", {}),
        ("cityblock", {}),
        ("chebyshev", {}),
        ("minkowski", {"p": 3}),
        ("braycurtis", {}),
        ("canberra", {}),
        ("hamming", {}),
        ("jaccard", {}),
        ("euclidean", {"w": np.arange(1, 11) / 4}),
    ],
)
def test_cdist_and_pdist_match_scipy_exactly(metric, kwargs):
    cross = distance.cdist(XA, XB, metric, **kwargs)
    pairs = distance.pdist(XA, metric, **kwargs)
    assert cross.shape == (6, 5) and cross.dtype == np.float64 and pairs.shape == (15,)
    assert_array_equal(cross.ravel()[:4], CDIST[metric, tuple(kwargs)])
    assert_array_equal(pairs[:4], PDIST[metric, tuple(kwargs)])


@pytest.mark.parametrize(
    "metric, kwargs",
    [("cosine", {}), ("correlation", {}), ("seuclidean", {}), ("mahalanobis", {"VI": VI})],
)
def test_cdist_and_pdist_agree_with_scipy_to_rounding(metric, kwargs):
    assert_allclose(distance.cdist(XA, XB, metric, **kwargs).ravel()[:4], CDIST[metric, tuple(kwargs)], rtol=1e-14)
    assert_allclose(distance.pdist(XA, metric, **kwargs)[:4], PDIST[metric, tuple(kwargs)], rtol=1e-14)


def test_pairwise_distances_agree_with_the_vector_functions():
    cross = distance.cdist(XA, XB, "cosine")
    assert_allclose(cross[2, 3], distance.cosine(XA[2], XB[3]), rtol=1e-14)
    square = distance.squareform(distance.pdist(XA, "cityblock"))
    assert square[1, 4] == square[4, 1]
    # The vector functions sum pairwise, as NumPy does, so they can round differently.
    assert_allclose(square[1, 4], distance.cityblock(XA[1], XA[4]), rtol=1e-15)


def test_metric_names_callables_and_out():
    X = np.array([[0.0, 0.0], [3.0, 4.0], [1.0, 1.0]])
    Y = np.array([[1.0, 0.0], [0.0, 2.0]])
    for alias in ["euclid", "EU", "e", "test_euclidean"]:
        assert_array_equal(distance.cdist(X, Y, alias), distance.cdist(X, Y))
    assert_array_equal(distance.cdist(X, Y, "co"), distance.cdist(X, Y, "correlation"))
    assert_array_equal(distance.pdist(X, "matching"), distance.pdist(X, "hamming"))
    assert_array_equal(
        distance.cdist(X, Y, lambda a, b: a.sum() - b.sum()), [[-1.0, -2.0], [6.0, 5.0], [1.0, 0.0]]
    )
    assert_array_equal(distance.pdist(X, lambda a, b, scale: scale * abs(a - b).sum(), scale=2), [14.0, 4.0, 10.0])
    out = np.zeros((3, 2))
    assert distance.cdist(X, Y, out=out) is out and out[1, 0] == np.sqrt(20.0)
    assert_array_equal(distance.pdist(np.array([[1, 2], [4, 6]])), [5.0])
    assert distance.pdist(np.zeros((1, 2))).shape == (0,)
    assert distance.cdist(np.zeros((0, 2)), Y).shape == (0, 2)


def test_seuclidean_and_mahalanobis_default_to_the_data_covariance():
    X = np.array([[0.0, 0.0], [3.0, 4.0], [1.0, 1.0]])
    Y = np.array([[1.0, 0.0], [0.0, 2.0]])
    assert_allclose(distance.cdist(X, Y, "seuclidean")[0], [0.816496580927726, 1.1952286093343936], rtol=1e-14)
    assert_allclose(distance.pdist(X, "seuclidean", V=[1, 2]), [4.123105625617661, 1.224744871391589, 2.9154759474226504])
    assert_allclose(distance.pdist(X, "mahalanobis", VI=np.eye(2) * 4), [10.0, 2.8284271247461903, 7.211102550927978])
    with pytest.raises(ValueError, match=r"^The number of observations \(2\) is too small; the covariance"):
        distance.pdist([[1.0, 2.0], [3.0, 4.0]], "mahalanobis")
    with pytest.raises(ValueError, match="^Variance vector V must be of the same dimension"):
        distance.pdist(X, "seuclidean", V=[1, 2, 3])


@pytest.mark.parametrize(
    "call, error, message",
    [
        (lambda: distance.cdist([[1.0]], [[1.0]], "nope"), ValueError, "Unknown Distance Metric: nope"),
        (lambda: distance.cdist([[1.0]], [[1.0]], 1), TypeError, "2nd argument metric must be a string identifier or a function."),
        (lambda: distance.pdist([1, 2, 3]), ValueError, "A 2-dimensional array must be passed. (Shape was (3,))."),
        (lambda: distance.cdist([1, 2], [[1, 2]]), ValueError, "XA must be a 2-dimensional array."),
        (lambda: distance.cdist([[1, 2]], [[1, 2, 3]]), ValueError, "XA and XB must have the same number of columns (i.e. feature dimension.)"),
        (lambda: distance.pdist(np.array([[1j, 2]])), ValueError, "Unsupported dtype complex128"),
        (lambda: distance.cdist([[1.0]], [[1.0]], out=np.zeros(2)), ValueError, "Output array has incorrect shape."),
        (lambda: distance.pdist([[1.0], [2.0]], out=np.zeros(1, dtype=np.float32)), ValueError, "wrong out dtype, expected float64"),
    ],
)
def test_pairwise_distance_errors(call, error, message):
    with pytest.raises(error) as raised:
        call()
    assert str(raised.value) == message


def test_squareform_converts_between_condensed_and_square_forms():
    condensed = np.array([1, 2, 3])
    square = distance.squareform(condensed)
    assert_array_equal(square, [[0, 1, 2], [1, 0, 3], [2, 3, 0]])
    assert square.dtype == np.int64
    assert_array_equal(distance.squareform(square), condensed)
    assert distance.squareform(np.array([1.5], dtype=np.float32)).dtype == np.float32
    assert_array_equal(distance.squareform([]), [[0.0]])
    assert distance.squareform(np.zeros((1, 1))).shape == (0,)
    assert_array_equal(distance.squareform(np.array([[0, 1], [2, 0]]), checks=False), [1])


@pytest.mark.parametrize(
    "argument, kwargs, message",
    [
        ([1, 2], {}, "Incompatible vector size. It must be a binomial coefficient n choose 2 for some integer n >= 2."),
        (np.ones((2, 2)), {}, "Distance matrix 'X' diagonal must be zero."),
        (np.array([[0, 1], [2, 0]]), {}, "Distance matrix 'X' must be symmetric."),
        (np.ones((2, 3)), {}, "The matrix argument must be square."),
        (np.zeros((2, 2, 2)), {}, "The first argument must be one or two dimensional array. A 3-dimensional array is not permitted"),
        ([1, 2, 3], {"force": "tovector"}, "Forcing 'tovector' but input X is not a distance matrix."),
        (np.zeros((3, 3)), {"force": "tomatrix"}, "Forcing 'tomatrix' but input X is not a distance vector."),
    ],
)
def test_squareform_errors(argument, kwargs, message):
    with pytest.raises(ValueError) as raised:
        distance.squareform(argument, **kwargs)
    assert str(raised.value) == message


def test_validity_checks_and_observation_counts():
    asymmetric = np.array([[0, 1], [2, 0]])
    assert distance.is_valid_dm(np.zeros((2, 2))) and not distance.is_valid_dm(asymmetric)
    with pytest.raises(ValueError, match="^Distance matrix 'D' must be symmetric within tolerance 0.10000.$"):
        distance.is_valid_dm(np.array([[0, 1], [1.5, 0]]), tol=0.1, throw=True)
    with pytest.raises(ValueError, match="^Distance matrix 'D' diagonal must be close to zero within tolerance 0.10000.$"):
        distance.is_valid_dm(np.array([[1, 1], [1, 0]]), tol=0.1, throw=True)
    with pytest.warns(UserWarning, match="^Distance matrix 'D' must be symmetric.$"):
        assert not distance.is_valid_dm(asymmetric, warning=True)
    assert distance.is_valid_y(np.zeros(3)) and not distance.is_valid_y(np.zeros(2))
    with pytest.raises(ValueError, match="^Condensed distance matrix must have shape=1"):
        distance.is_valid_y(np.zeros((2, 2)), throw=True)
    assert (distance.num_obs_y([1, 2, 3]), distance.num_obs_y([1]), distance.num_obs_dm(np.zeros((3, 3)))) == (3, 2, 3)
    with pytest.raises(ValueError, match="^The number of observations cannot be determined on an empty distance matrix.$"):
        distance.num_obs_y([])


def test_deprecated_minkowski_helpers():
    X = np.array([[0.0, 0.0], [3.0, 4.0]])
    Y = np.array([[1.0, 0.0], [0.0, 2.0]])
    with pytest.warns(DeprecationWarning, match="`distance_matrix` is deprecated in favor of `scipy.spatial.distance.cdist`"):
        assert_array_equal(scipy.spatial.distance_matrix(X, Y, p=1), [[1.0, 2.0], [6.0, 5.0]])
    with pytest.warns(DeprecationWarning, match="`minkowski_distance` is deprecated"):
        assert_allclose(scipy.spatial.minkowski_distance(X, Y, p=3), [1.0, 35.0 ** (1 / 3)], rtol=1e-15)
    with pytest.warns(DeprecationWarning, match="`minkowski_distance_p` is deprecated"):
        assert_array_equal(scipy.spatial.minkowski_distance_p(X, Y, p=3), [1.0, 35.0])


# The first four entries of cdist(XA, XB) and pdist(XA), measured with SciPy 1.18.1.
CDIST = {
    ('euclidean', ()): [3.415650255319866, 3.651483716701107, 4.013864859597431, 4.384315479321969],
    ('sqeuclidean', ()): [11.666666666666664, 13.333333333333332, 16.111111111111107, 19.222222222222225],
    ('cityblock', ()): [9.0, 9.333333333333332, 11.000000000000002, 11.000000000000002],
    ('chebyshev', ()): [2.0, 2.0, 2.0, 2.666666666666667],
    ('minkowski', ('p',)): [2.6044630773718573, 2.7507905083798105, 3.008207975241695, 3.404044910409521],
    ('braycurtis', ()): [0.6585365853658537, 0.7777777777777778, 1.137931034482759, 1.0645161290322585],
    ('canberra', ()): [6.383333333333334, 6.476190476190476, 8.392857142857142, 6.8],
    ('hamming', ()): [0.9, 0.8, 1.0, 0.8],
    ('jaccard', ()): [0.3, 0.2, 0.3, 0.2],
    ('euclidean', ('w',)): [3.8042374035044424, 4.722875771768251, 4.226240777701989, 4.0858835573770875],
    ('cosine', ()): [0.6469405471260374, 0.7952946273037474, 0.9844620058815026, 1.1625753849405067],
    ('correlation', ()): [0.6467784817271538, 0.7933458302282081, 0.9812468864316582, 1.17187006012718],
    ('seuclidean', ()): [3.640027137808519, 3.8328774581165916, 4.377523184940733, 4.617208323519306],
    ('mahalanobis', ('VI',)): [3.41797303712883, 3.660167400109645, 4.031621045431757, 4.4005771627230175],
}
PDIST = {
    ('euclidean', ()): [5.696002496878354, 5.322906474223771, 3.48010216963685, 5.821416398857661],
    ('sqeuclidean', ()): [32.44444444444444, 28.333333333333336, 12.11111111111111, 33.88888888888889],
    ('cityblock', ()): [17.333333333333336, 15.0, 6.333333333333333, 18.333333333333332],
    ('chebyshev', ()): [2.3333333333333335, 2.666666666666667, 3.3333333333333335, 2.0],
    ('minkowski', ('p',)): [4.021489284197725, 3.9976838442997362, 3.3433034824392323, 3.9821737607333576],
    ('braycurtis', ()): [1.7333333333333332, 1.4516129032258063, 0.3877551020408163, 2.0370370370370368],
    ('canberra', ()): [9.333333333333334, 8.457142857142857, 3.574603174603175, 10.0],
    ('hamming', ()): [1.0, 1.0, 1.0, 1.0],
    ('jaccard', ()): [0.2, 0.2, 0.1, 0.2],
    ('euclidean', ('w',)): [6.747427493332387, 6.180165405913052, 3.5394600969325505, 6.837397165588672],
    ('cosine', ()): [1.3907598379778476, 1.25685981900858, 0.4977116585250546, 1.5038404142091375],
    ('correlation', ()): [1.389819383765292, 1.2631806779839077, 0.49748109237039395, 1.5044296328024895],
    ('seuclidean', ()): [5.1195089095784265, 4.51309361234806, 2.86841565463359, 5.185313922028653],
    ('mahalanobis', ('VI',)): [5.718252591344028, 5.3363086938623105, 3.4823819616728042, 5.855400437691198],
}
