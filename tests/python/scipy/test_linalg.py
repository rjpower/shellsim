# Portable SciPy semantics. Expectations checked against SciPy 1.18.1 and NumPy 2.5.3 on
# CPython 3.14.4.
# Scope: scipy.linalg solvers, decompositions, matrix functions, special matrices, and the BLAS
# and LAPACK wrappers.
# LU, Cholesky, triangular, QR and matrix-function results are compared exactly. They were
# measured with SciPy's OpenBLAS on an AMD Zen 2 machine, where OpenBLAS selects its Haswell
# kernels and runs 16 threads; other kernels can round the order-12 and order-70 cases
# differently. Eigenvalue, singular value and least-squares results come from different
# algorithms in shellsim and are compared to 1e-13 relative, with eigenvectors up to sign.

import warnings

import numpy as np
import pytest
import scipy.linalg as sl
from numpy.testing import assert_allclose, assert_array_equal
from scipy.linalg import blas, lapack

G = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 10.0]])
S = np.array([[4.0, 1.0, 2.0], [1.0, 5.0, 3.0], [2.0, 3.0, 6.0]])
b = np.array([1.0, 2.0, 3.0])
R = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])


def close(actual, expected, rtol=1e-13, atol=1e-14):
    assert_allclose(actual, expected, rtol=rtol, atol=atol)


def signed_columns(vectors):
    """Columns scaled so that each first element is positive, as eigenvector signs are free."""
    return vectors * np.sign(vectors[0])


def recorded(function, *args, **kwargs):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = function(*args, **kwargs)
    return result, [(warning.category.__name__, str(warning.message)) for warning in caught]


def order_12():
    n = 12
    a = ((np.arange(n * n).reshape(n, n) * 37) % 23 - 11) / 7.0 + n * np.eye(n)
    return a, np.arange(1, n + 1) / 3.0


def order_70_lower_triangle():
    k = 70
    return np.tril(((np.arange(k * k).reshape(k, k) * 13) % 17 - 8) / 5.0) + 3 * np.eye(k)


def test_solve_rounds_as_scipy_does():
    assert_array_equal(sl.solve(G, b), [-0.3333333333333333, 0.6666666666666666, -0.0])
    assert_array_equal(
        sl.solve(G, np.array([[1.0, 0.0], [2.0, 1.0], [3.0, 0.0]])),
        [[-0.33333333333333315, -1.333333333333333], [0.6666666666666665, 3.6666666666666665], [-0.0, -2.0]],
    )
    assert_array_equal(sl.solve(G, b, transposed=True), [1.0, -0.0, 0.0])
    positive = [-2.0816681711721685e-17, 0.14285714285714277, 0.42857142857142866]
    assert_array_equal(sl.solve(S, b), positive)
    assert_array_equal(sl.solve(S, b, assume_a="pos"), positive)
    assert_array_equal(sl.solve(S, b, assume_a="sym"), [0.0, 0.14285714285714285, 0.4285714285714286])
    assert_array_equal(
        sl.solve(np.tril(S), b, assume_a="sym", lower=True), [0.0, 0.14285714285714285, 0.42857142857142855]
    )
    assert_array_equal(
        sl.solve(S, b, assume_a="gen"), [-6.938893903907228e-18, 0.14285714285714282, 0.4285714285714286]
    )
    assert_array_equal(sl.solve(np.diag([2.0, 4.0, 8.0]), b), [0.5, 0.5, 0.375])
    assert_array_equal(sl.solve(np.diag([2.0, 4.0, 8.0]), b, assume_a="diagonal"), [0.5, 0.5, 0.375])
    tridiagonal = np.array([[4.0, 1, 0, 0], [1, 4, 1, 0], [0, 1, 4, 1], [0, 0, 1, 4]])
    assert_array_equal(
        sl.solve(tridiagonal, np.array([1.0, 2, 3, 4])),
        [0.1626794258373206, 0.3492822966507177, 0.4401913875598086, 0.8899521531100478],
    )
    upper = [0.020000000000000018, 0.040000000000000036, 0.3]
    assert_array_equal(sl.solve(np.triu(G), b), upper)
    assert_array_equal(sl.solve(np.triu(G), b, assume_a="upper triangular"), upper)
    assert_array_equal(
        sl.solve(np.tril(G), b, assume_a="lower triangular"), [1.0, -0.4, -0.07999999999999999]
    )


def test_solve_casts_integers_and_keeps_float32():
    result = sl.solve(np.array([[2, 1], [1, 3]]), np.array([1, 2]))
    assert result.dtype == np.float64
    assert_array_equal(result, [0.19999999999999998, 0.6])
    single = sl.solve(G.astype(np.float32), b.astype(np.float32))
    assert single.dtype == np.float32
    assert_array_equal(single, np.array([-0.3333333432674408, 0.6666666865348816, -0.0], dtype=np.float32))


def test_solve_batches_over_leading_dimensions():
    result = sl.solve(np.stack([G, S]), np.stack([b, b])[..., None])[..., 0]
    assert_array_equal(
        result,
        [
            [-0.3333333333333333, 0.6666666666666666, -0.0],
            [-2.0816681711721685e-17, 0.14285714285714277, 0.42857142857142866],
        ],
    )


def test_order_12_solves_round_as_scipy_does():
    a, rhs = order_12()
    assert_array_equal(
        sl.solve(a, rhs),
        [
            0.037487037353417615, 0.08425926930020734, 0.10024115150536743, 0.10804276874165834,
            0.11703376798584372, 0.2052026780084801, 0.1660748569156105, 0.2244571157105786,
            0.2667378514596518, 0.27563436174075834, 0.2884282384053866, 0.3806733478409745,
        ],
    )
    assert_array_equal(
        sl.inv(a)[0],
        [
            0.09869745199551529, -0.002539431157618714, 0.004826685863455456,
            -0.010937582060045426, 0.0012987027723003431, 0.014175614957781841,
            -0.005843843994083962, 0.004731125032049925, -0.01083065867233664,
            0.003991617449197632, 0.009992297117893095, -0.007727668134498731,
        ],
    )
    assert sl.det(a) == 8048664735173.506
    lu, piv = sl.lu_factor(a)
    assert_array_equal(piv, np.arange(12))
    assert_array_equal(
        lu[-1],
        [
            -0.0410958904109589, 0.11812627291242361, 0.018139888998002847, -0.0646886362640402,
            0.07130014148356346, -0.01091688516793359, -0.15001346280014924, 0.011737006189006614,
            -0.04118274277799949, 0.08275002634929571, -0.0016069978962442737, 10.627544720200753,
        ],
    )
    spd = a + a.T + 48 * np.eye(12)
    assert_array_equal(
        sl.cholesky(spd)[:, -1],
        [
            0.03443161999160073, 6.85123956611605e-05, -0.03294480101371222, -0.06842816329103996,
            0.2823351903490355, -0.1355311138886872, -0.17097881850502525, 0.18628444485078566,
            -0.23809938681564627, 0.12622122276744574, 0.10020276593293177, 8.300073651816058,
        ],
    )
    assert_array_equal(
        sl.solve(spd, rhs, assume_a="pos"),
        [
            0.005820362882250528, 0.008963493812397307, 0.014011382524806947,
            0.018142377064649395, 0.02130144509743144, 0.028948755334889695,
            0.03267253042702954, 0.03806250766741902, 0.04250927745785542,
            0.04569214753474603, 0.05077483114198197, 0.057619556693082684,
        ],
    )
    assert_array_equal(
        sl.inv(spd, assume_a="pos")[0],
        [
            0.014599676548413616, 1.4433788368471004e-05, 6.175105678814119e-05,
            -0.0005035444055723404, 0.00019304650274469925, 0.0002681071668185255,
            -0.0003648593077899985, 0.0003887439444771287, -0.00024192486694195054,
            -0.00017682954871000285, 0.0006000403318784634, -9.35972752901033e-05,
        ],
    )


def test_order_70_triangular_solves_round_as_scipy_does():
    t = order_70_lower_triangle()
    ones = np.ones(70)
    assert_array_equal(
        sl.solve_triangular(t, ones, lower=True)[-3:],
        [30.475696929792843, 135.68244809223194, 36.92835105526551],
    )
    assert_array_equal(
        sl.solve_triangular(t, ones, lower=True, trans="T")[:3],
        [189.26017581550266, -22.53851361570636, -8.99433637908656],
    )
    assert_array_equal(sl.inv(t)[-1, :3], [40.97633909468525, -3.116640968016121, -2.2612036346198114])
    # SciPy's threaded OpenBLAS solves each of several right-hand sides separately.
    columns = np.stack([np.arange(1.0, 71), ones], axis=1)
    assert_array_equal(
        sl.solve_triangular(t, columns, lower=True)[-2:],
        [[153.87403273374383, 135.68244809223185], [52.634174722032675, 36.92835105526547]],
    )


def test_inv_and_det_round_as_scipy_does():
    assert_array_equal(
        sl.inv(G),
        [
            [-0.6666666666666662, -1.333333333333333, 0.9999999999999996],
            [-0.6666666666666676, 3.6666666666666665, -1.9999999999999996],
            [1.0000000000000004, -2.0, 0.9999999999999999],
        ],
    )
    assert_array_equal(
        sl.inv(S, assume_a="pos"),
        [
            [0.3, 1.3877787807814457e-17, -0.10000000000000002],
            [1.3877787807814457e-17, 0.28571428571428564, -0.14285714285714285],
            [-0.10000000000000002, -0.14285714285714285, 0.27142857142857146],
        ],
    )
    assert_array_equal(
        sl.inv(np.triu(G)), [[1.0, -0.4, -0.05999999999999997], [0.0, 0.2, -0.12000000000000002], [0.0, 0.0, 0.1]]
    )
    assert_array_equal(
        sl.inv(np.array([[2, 1], [1, 3]])), [[0.5999999999999999, -0.19999999999999998], [-0.19999999999999998, 0.4]]
    )
    single = sl.inv(G.astype(np.float32))
    assert single.dtype == np.float32
    assert_array_equal(
        single,
        np.array(
            [
                [-0.6666670441627502, -1.3333334922790527, 1.000000238418579],
                [-0.6666659116744995, 3.6666667461395264, -2.000000238418579],
                [0.9999995827674866, -2.0, 1.0000001192092896],
            ],
            dtype=np.float32,
        ),
    )
    det = sl.det(G)
    assert type(det) is np.float64 and det == -3.0
    assert sl.det(S.astype(np.float32)) == 70.0
    assert sl.det(G.astype(np.float32)) == -2.999999761581421
    assert sl.det(np.array([[2, 1], [1, 3]])) == 5.0
    assert_array_equal(sl.det(np.stack([G, S])), [-3.0, 70.0])
    assert_array_equal(sl.inv(np.stack([np.eye(2), 2 * np.eye(2)])), [np.eye(2), 0.5 * np.eye(2)])
    assert sl.inv(np.empty((0, 0))).shape == (0, 0)
    assert sl.det(np.empty((0, 0))) == 1.0


def test_float16_and_bool_input_is_deprecated():
    result, caught = recorded(
        sl.solve, np.array([[2.0, 1.0], [1.0, 3.0]], dtype=np.float16), np.array([1.0, 2.0], dtype=np.float16)
    )
    assert result.dtype == np.float32
    assert_array_equal(result, np.array([0.20000000298023224, 0.5999999642372131], dtype=np.float32))
    assert caught == [
        (
            "DeprecationWarning",
            "Calling linalg.solve with arguments of dtype=float16 (a.dtype.char = 'e') is deprecated "
            "in SciPy 1.18.0 and will be removed in SciPy 1.20.0. Please cast array inputs to one of "
            "np.float{32,64} or np.complex{64,128} manually.",
        )
    ]
    result, caught = recorded(sl.det, np.array([[True, False], [False, True]]))
    assert result == 1.0
    assert caught[0][0] == "DeprecationWarning" and "dtype=bool (a.dtype.char = '?')" in caught[0][1]


def test_lu_decompositions_round_as_scipy_does():
    p, l, u = sl.lu(G)
    assert_array_equal(p, [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])
    assert_array_equal(l, [[1.0, 0.0, 0.0], [0.14285714285714285, 1.0, 0.0], [0.5714285714285714, 0.5000000000000002, 1.0]])
    assert_array_equal(u, [[7.0, 8.0, 10.0], [0.0, 0.8571428571428572, 1.5714285714285716], [0.0, 0.0, -0.5]])
    permuted, _ = sl.lu(G, permute_l=True)
    assert_array_equal(
        permuted, [[0.14285714285714285, 1.0, 0.0], [0.5714285714285714, 0.5000000000000002, 1.0], [1.0, 0.0, 0.0]]
    )
    indices, _, _ = sl.lu(G, p_indices=True)
    assert indices.dtype == np.int32
    assert_array_equal(indices, [1, 2, 0])
    lu, piv = sl.lu_factor(G)
    assert piv.dtype == np.int32
    assert_array_equal(piv, [2, 2, 2])
    assert_array_equal(
        lu, [[7.0, 8.0, 10.0], [0.14285714285714285, 0.8571428571428572, 1.5714285714285716], [0.5714285714285714, 0.5000000000000002, -0.5]]
    )
    assert_array_equal(sl.lu_solve((lu, piv), b), [-0.3333333333333333, 0.6666666666666666, -0.0])
    assert_array_equal(sl.lu_solve((lu, piv), b, trans=1), [1.0, -0.0, 0.0])
    assert_array_equal(sl.lu_solve((lu, piv), b, trans=2), [1.0, -0.0, 0.0])
    _, l, u = sl.lu(R)
    assert_array_equal(l, [[1.0, 0.0], [0.2, 1.0], [0.6000000000000001, 0.49999999999999944]])
    assert_array_equal(u, [[5.0, 6.0], [0.0, 0.7999999999999998]])
    p, l, u = sl.lu(np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))
    assert_array_equal(p, [[0.0, 1.0], [1.0, 0.0]])
    assert_array_equal(l, [[1.0, 0.0], [0.25, 1.0]])
    assert_array_equal(u, [[4.0, 5.0, 6.0], [0.0, 0.75, 1.5]])
    assert_array_equal(
        sl.lu(np.stack([G, S]))[2][1], [[4.0, 1.0, 2.0], [0.0, 4.75, 2.5], [0.0, 0.0, 3.6842105263157894]]
    )
    single, _ = sl.lu_factor(G.astype(np.float32))
    assert_array_equal(
        single,
        np.array(
            [[7.0, 8.0, 10.0], [0.1428571492433548, 0.8571428060531616, 1.5714285373687744], [0.5714285969734192, 0.4999997913837433, -0.5]],
            dtype=np.float32,
        ),
    )


def test_cholesky_decompositions_round_as_scipy_does():
    upper = [[2.0, 0.5, 1.0], [0.0, 2.179449471770337, 1.1470786693528088], [0.0, 0.0, 1.9194297398747862]]
    assert_array_equal(sl.cholesky(S), upper)
    assert_array_equal(sl.cholesky(S, lower=True), np.array(upper).T)
    c, lower = sl.cho_factor(S)
    assert_array_equal(c, upper)
    assert isinstance(lower, np.ndarray) and lower.shape == () and not lower
    c_lower, lower = sl.cho_factor(S, lower=True)
    assert_array_equal(c_lower, np.array(upper).T)
    expected = [-2.0816681711721685e-17, 0.14285714285714277, 0.42857142857142866]
    assert_array_equal(sl.cho_solve((c, False), b), expected)
    assert_array_equal(sl.cho_solve((c_lower, True), b), expected)
    assert_array_equal(
        sl.cholesky(S.astype(np.float32)),
        np.array(
            [[2.0, 0.5, 1.0], [0.0, 2.1794495582580566, 1.1470786333084106], [0.0, 0.0, 1.9194297790527344]],
            dtype=np.float32,
        ),
    )


def test_triangular_banded_and_circulant_solves():
    assert_array_equal(sl.solve_triangular(np.tril(G), b, lower=True), [1.0, -0.4, -0.07999999999999999])
    assert_array_equal(sl.solve_triangular(np.triu(G), b, trans="T"), [1.0, 0.0, 0.0])
    assert_array_equal(sl.solve_triangular(np.tril(G), b, lower=True, unit_diagonal=True), [1.0, -2.0, 12.0])
    bands = np.array([[0.0, 1.0, 2.0], [4.0, 5.0, 6.0], [7.0, 8.0, 0.0]])
    assert_array_equal(sl.solve_banded((1, 1), bands, b), [0.5714285714285718, -1.285714285714287, 2.2142857142857157])
    assert_array_equal(
        sl.solve_banded((1, 1), bands, np.stack([b, 2 * b], axis=1)),
        [[0.5714285714285718, 1.1428571428571437], [-1.285714285714287, -2.571428571428574], [2.2142857142857157, 4.4285714285714315]],
    )
    wide = np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 2.0], [4.0, 5.0, 6.0], [7.0, 8.0, 0.0]])
    assert_array_equal(sl.solve_banded((1, 2), wide, b), [0.12857142857142861, 0.042857142857142816, 0.4428571428571429])
    assert_array_equal(sl.solve_banded((0, 1), np.array([[0.0, 1.0, 2.0], [4.0, 5.0, 6.0]]), b), [0.2, 0.2, 0.5])
    # solve_circulant goes through an FFT/IFFT round trip, so its near-zero entries land at
    # FFT-implementation-dependent rounding noise (order 1e-16) rather than an exact zero.
    assert_allclose(sl.solve_circulant(np.array([2.0, 1.0, 0.0]), b), [0.0, 1.0, 1.0], atol=1e-12)
    assert_allclose(
        sl.solve_circulant(np.array([2.0, 1.0, 0.0]), np.stack([b, b], axis=1)),
        [[0.0, 0.0], [1.0, 1.0], [1.0, 1.0]],
        atol=1e-12,
    )


def test_qr_rounds_as_scipy_does():
    q, r = sl.qr(G)
    assert_array_equal(
        r,
        [[-8.124038404635959, -9.601136296387955, -11.939874624995277], [0.0, 0.9045340337332926, 1.50755672288882], [0.0, 0.0, 0.40824829046386224]],
    )
    assert_array_equal(
        q,
        [
            [-0.12309149097933281, 0.9045340337332914, 0.4082482904638621],
            [-0.492365963917331, 0.30151134457776285, -0.8164965809277264],
            [-0.8616404368553292, -0.3015113445777631, 0.4082482904638634],
        ],
    )
    (r_only,) = sl.qr(G, mode="r")
    assert_array_equal(r_only, r)
    q, r = sl.qr(R, mode="economic")
    assert_array_equal(r, [[-5.916079783099616, -7.437357441610946], [0.0, 0.828078671210825]])
    assert_array_equal(
        q,
        [[-0.16903085094570325, 0.8970852271450607], [-0.50709255283711, 0.27602622373694136], [-0.8451542547285166, -0.34503277967117696]],
    )
    assert [x.shape for x in sl.qr(R)] == [(3, 3), (3, 2)]
    _, r, permutation = sl.qr(G, pivoting=True)
    assert permutation.dtype == np.int32
    assert_array_equal(permutation, [2, 0, 1])
    assert_array_equal(
        r,
        [
            [-12.041594578792296, -8.055411545812776, -9.633275663033835],
            [0.0, -1.0537290105080197, -0.37960423981034364],
            [0.0, 0.0, -0.23643312187173038],
        ],
    )


def test_eigh_agrees_with_scipy():
    w, v = sl.eigh(S)
    close(w, [2.1943971674224088, 3.3867701566075468, 9.418832675970034])
    close(
        signed_columns(v),
        [
            [0.44122469687346455, 0.8155834192895001, 0.3743587224160365],
            [0.5773502691896257, -0.5773502691896262, 0.577350269189625],
            [-0.6870134158337711, 0.038605088352622496, 0.7256185041863941],
        ],
    )
    generalized = [0.9183340005338673, 2.5000000000000004, 5.081665999466133]
    close(sl.eigvalsh(S, np.diag([1.0, 2.0, 3.0])), generalized)
    close(sl.eigh(S, np.diag([1.0, 2.0, 3.0]))[0], generalized)
    close(sl.eigh(S, eigvals_only=True, subset_by_index=[1, 2]), [3.3867701566075468, 9.418832675970037])
    close(sl.eigh(S, eigvals_only=True, subset_by_value=(3.0, 10.0)), [3.3867701566075485, 9.41883267597004])
    close(sl.eigvalsh(np.triu(S), lower=False), [2.1943971674224065, 3.38677015660755, 9.41883267597004])


def test_svd_agrees_with_scipy():
    u, s, vh = sl.svd(G)
    close(s, [17.412505166808593, 0.8751613501104364, 0.19686652111743])
    close(
        signed_columns(u),
        [
            [0.20933734352577438, 0.9643851351290611, 0.16167618175747142],
            [0.5038485108077233, 0.03532145172128193, -0.863069564522548],
            [0.8380420960562849, -0.2621329933266777, 0.4785099153070752],
        ],
    )
    close(
        signed_columns(vh.T).T,
        [
            [0.46466754677974925, 0.5537545526717395, 0.6909703078751004],
            [0.8332863547739321, -0.009499485152154541, -0.552759993785202],
            [0.299529500913119, -0.8326257593764232, 0.46585021509662555],
        ],
    )
    close(sl.svdvals(G), [17.412505166808597, 0.8751613501104364, 0.19686652111742997])
    assert [x.shape for x in sl.svd(R, full_matrices=False)] == [(3, 2), (2,), (2, 2)]
    assert [x.shape for x in sl.svd(R)] == [(3, 3), (2,), (2, 2)]
    close(sl.svd(R, compute_uv=False), [9.525518091565107, 0.5143005806586443])
    assert_array_equal(sl.diagsvd([1.0, 2.0], 3, 2), [[1.0, 0.0], [0.0, 2.0], [0.0, 0.0]])


def test_least_squares_and_pseudo_inverses_agree_with_scipy():
    x, residues, rank, singular = sl.lstsq(np.array([[1.0, 1.0], [1.0, 2.0], [1.0, 3.0]]), np.array([1.0, 2.0, 2.0]))
    close(x, [0.6666666666666663, 0.5000000000000002])
    assert residues.shape == ()
    close(residues, 0.16666666666666677)
    assert type(rank) is np.int64 and rank == 2
    close(singular, [4.079143328941734, 0.6004912172131635])
    x, residues, rank, singular = sl.lstsq(np.array([[1.0, 2.0], [2.0, 4.0], [3.0, 6.0]]), b)
    close(x, [0.19999999999999996, 0.39999999999999997])
    assert np.isnan(residues) and rank == 1
    close(singular, [8.366600265340756, 8.881784197001251e-16])
    x, residues, rank, _ = sl.lstsq(R, np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]))
    close(x, [[-0.666666666666667, 0.33333333333333376], [0.6666666666666669, -0.08333333333333358]])
    close(residues, [0.6666666666666666, 0.16666666666666657])
    x, residues, rank, singular = sl.lstsq(R, b, lapack_driver="gelsy")
    close(x, [-1.4430967688835235e-15, 0.5000000000000012])
    assert residues.shape == (0,) and rank == 2 and singular is None
    x, residues, rank, _ = sl.lstsq(np.array([[1.0, 2.0, 3.0]]), np.array([1.0]))
    close(x, [0.0714285714285714, 0.14285714285714285, 0.2142857142857143])
    assert residues.shape == (0,) and rank == 1
    close(
        sl.pinv(R),
        [[-1.3333333333333337, -0.33333333333333287, 0.6666666666666665], [1.0833333333333335, 0.3333333333333329, -0.4166666666666665]],
    )
    assert sl.pinv(np.array([[1.0, 2.0], [2.0, 4.0], [3.0, 6.0]]), return_rank=True)[1] == 1
    close(
        sl.pinvh(S),
        [
            [0.3000000000000001, -3.316498068850119e-17, -0.09999999999999999],
            [-1.4222414916514175e-17, 0.28571428571428575, -0.1428571428571428],
            [-0.09999999999999999, -0.1428571428571428, 0.2714285714285714],
        ],
    )


def test_subspaces_polar_and_procrustes_agree_with_scipy():
    assert sl.null_space(np.array([[1.0, 1.0, 1.0]])).shape == (3, 2)
    close(np.abs(sl.null_space(np.array([[1.0, 1.0]]))), [[0.7071067811865475], [0.7071067811865475]])
    close(np.abs(sl.orth(np.array([[1.0, 1.0], [1.0, 1.0]]))), [[0.7071067811865472], [0.7071067811865475]])
    close(
        sl.subspace_angles(np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]]), np.array([[1.0], [1.0], [1.0]])),
        [0.6154797086703873],
    )
    u, p = sl.polar(np.array([[1.0, 2.0], [3.0, 4.0]]))
    close(u, [[-0.5144957554275261, 0.8574929257125442], [0.8574929257125442, 0.5144957554275263]])
    close(p, [[2.0579830217101063, 2.400980191995124], [2.400980191995124, 3.772968873135193]])
    rotation, scale = sl.orthogonal_procrustes(np.eye(2), np.array([[0.0, 1.0], [-1.0, 0.0]]))
    close(rotation, [[0.0, 1.0], [-1.0, 0.0]])
    assert scale == 2.0
    close(sl.orthogonal_procrustes(R, R[:, ::-1])[1], 91.0)


def test_matrix_functions_round_as_scipy_does():
    assert_array_equal(
        sl.expm(np.array([[0.1, 0.2], [0.3, 1.9]])),
        [[1.1720433769251966, 0.6259875479854236], [0.9389813219781356, 6.805931308794008]],
    )
    assert_array_equal(sl.expm(np.zeros((2, 2))), np.eye(2))
    assert_array_equal(
        sl.expm(np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [-1.0, -2.0, -3.0]])),
        [
            [0.9165682455933042, 0.8091961631129424, 0.19658289936556927],
            [-0.19658289936556927, 0.5234024468621656, 0.2194474650162344],
            [-0.21944746501623455, -0.6354778293980385, -0.13493994818653787],
        ],
    )
    assert_array_equal(
        sl.expm(np.array([[10.0, 3.0], [2.0, -4.0]])),
        [[32459.13503668736, 6754.718933498614], [4503.145955665742, 937.1133470271582]],
    )
    assert_array_equal(
        sl.expm(np.stack([np.zeros((2, 2)), np.eye(2)])), [np.eye(2), 2.718281828459045 * np.eye(2)]
    )
    single = sl.expm(np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32))
    assert single.dtype == np.float32
    assert_array_equal(
        single, np.array([[1.142093539237976, 0.2603507339954376], [0.3905261754989624, 1.5326197147369385]], dtype=np.float32)
    )
    m = np.array([[0.1, 0.2], [0.3, 0.4]])
    assert_array_equal(sl.coshm(m), [[1.0358371842965521, 0.05122001843907957], [0.07683002765861932, 1.1126672119551715]])
    assert_array_equal(sl.sinhm(m), [[0.10625636462302057, 0.20913072910161318], [0.3136960936524197, 0.41995245827544037]])
    assert_array_equal(sl.tanhm(m), [[0.08894292636419815, 0.1838600693676731], [0.27579010405150955, 0.3647330304157078]])


def test_norms_and_structure_checks():
    assert sl.norm(b) == 3.7416573867739413
    assert sl.norm(G) == 17.435595774162696
    assert sl.norm(G, "fro") == 17.435595774162696
    assert (sl.norm(G, 1), sl.norm(G, np.inf), sl.norm(G, -1), sl.norm(G, -np.inf)) == (19.0, 25.0, 12.0, 6.0)
    assert (sl.norm(b, 1), sl.norm(b, np.inf), sl.norm(b, 0)) == (6.0, 3.0, 3.0)
    assert sl.norm(b, 3) == 3.3019272488946263
    close([sl.norm(G, 2), sl.norm(G, -2), sl.norm(G, "nuc")], [17.412505166808597, 0.19686652111742997, 18.48453303803646])
    assert_array_equal(sl.norm(G, axis=0), [8.12403840463596, 9.643650760992955, 12.041594578792296])
    assert_array_equal(
        sl.norm(G, axis=1, keepdims=True), [[3.7416573867739413], [8.774964387392123], [14.594519519326424]]
    )
    assert sl.bandwidth(np.triu(G)) == (0, 2)
    assert sl.bandwidth(G) == (2, 2)
    assert sl.bandwidth(np.eye(3)) == (0, 0)
    assert sl.issymmetric(S) and not sl.issymmetric(G) and sl.ishermitian(S)
    assert sl.issymmetric(S + 1e-12 * G, rtol=1e-10)
    assert sl.issymmetric(np.array([[1.0, 2.0], [2.0 + 1e-9, 1.0]]), atol=1e-8)


def test_special_matrices():
    cases = [
        (sl.toeplitz([1, 2, 3], [1, 4, 5]), [[1, 4, 5], [2, 1, 4], [3, 2, 1]], np.int64),
        (sl.toeplitz([1.0, 2.0, 3.0]), [[1.0, 2.0, 3.0], [2.0, 1.0, 2.0], [3.0, 2.0, 1.0]], np.float64),
        (sl.circulant([1, 2, 3]), [[1, 3, 2], [2, 1, 3], [3, 2, 1]], np.int64),
        (sl.hankel([1, 2, 3], [3, 4, 5]), [[1, 2, 3], [2, 3, 4], [3, 4, 5]], np.int64),
        (sl.hadamard(4), [[1, 1, 1, 1], [1, -1, 1, -1], [1, 1, -1, -1], [1, -1, -1, 1]], np.int64),
        (
            sl.leslie([0.1, 2.0, 1.0, 0.1], [0.2, 0.8, 0.7]),
            [[0.1, 2.0, 1.0, 0.1], [0.2, 0.0, 0.0, 0.0], [0.0, 0.8, 0.0, 0.0], [0.0, 0.0, 0.7, 0.0]],
            np.float64,
        ),
        (sl.block_diag([[1, 2]], [[3], [4]], 5), [[1, 2, 0, 0], [0, 0, 3, 0], [0, 0, 4, 0], [0, 0, 0, 5]], np.int64),
        (sl.companion([1, -10, 31, -30]), [[10.0, -31.0, 30.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], np.float64),
        (sl.companion([2.0, 4.0, 6.0]), [[-2.0, -3.0], [1.0, 0.0]], np.float64),
        (
            sl.helmert(3),
            [[0.7071067811865475, -0.7071067811865475, 0.0], [0.4082482904638631, 0.4082482904638631, -0.8164965809277261]],
            np.float64,
        ),
        (
            sl.helmert(3, full=True),
            [
                [0.5773502691896258, 0.5773502691896258, 0.5773502691896258],
                [0.7071067811865475, -0.7071067811865475, 0.0],
                [0.4082482904638631, 0.4082482904638631, -0.8164965809277261],
            ],
            np.float64,
        ),
        (
            sl.hilbert(3),
            [[1.0, 0.5, 0.3333333333333333], [0.5, 0.3333333333333333, 0.25], [0.3333333333333333, 0.25, 0.2]],
            np.float64,
        ),
        (sl.invhilbert(3), [[9.0, -36.0, 30.0], [-36.0, 192.0, -180.0], [30.0, -180.0, 180.0]], np.float64),
        (
            sl.invhilbert(4, exact=True),
            [[16, -120, 240, -140], [-120, 1200, -2700, 1680], [240, -2700, 6480, -4200], [-140, 1680, -4200, 2800]],
            np.int64,
        ),
        (sl.pascal(4), [[1, 1, 1, 1], [1, 2, 3, 4], [1, 3, 6, 10], [1, 4, 10, 20]], np.uint64),
        (sl.pascal(4, kind="lower"), [[1, 0, 0, 0], [1, 1, 0, 0], [1, 2, 1, 0], [1, 3, 3, 1]], np.uint64),
        (sl.pascal(3, kind="upper"), [[1, 1, 1], [0, 1, 2], [0, 0, 1]], np.uint64),
        (sl.invpascal(4), [[4, -6, 4, -1], [-6, 14, -11, 3], [4, -11, 10, -3], [-1, 3, -3, 1]], np.int64),
        (sl.invpascal(3, exact=True), [[3, -3, 1], [-3, 5, -2], [1, -2, 1]], np.int64),
        (sl.fiedler([1, 4, 12, 45]), [[0, 3, 11, 44], [3, 0, 8, 41], [11, 8, 0, 33], [44, 41, 33, 0]], np.int64),
        (sl.fiedler_companion([1, -10, 31, -30]), [[10.0, -31.0, 1.0], [1.0, 0.0, 0.0], [0.0, 30.0, 0.0]], np.float64),
        (
            sl.convolution_matrix([1, 2, 3], 4, mode="same"),
            [[2, 1, 0, 0], [3, 2, 1, 0], [0, 3, 2, 1], [0, 0, 3, 2]],
            np.int64,
        ),
        (sl.convolution_matrix([1, 2], 3), [[1, 0, 0], [2, 1, 0], [0, 2, 1], [0, 0, 2]], np.int64),
        (sl.convolution_matrix([1, 2], 3, mode="valid"), [[2, 1, 0], [0, 2, 1]], np.int64),
        (
            sl.khatri_rao(np.array([[1, 2], [3, 4]]), np.array([[5, 6], [7, 8], [9, 10]])),
            [[5, 12], [7, 16], [9, 20], [15, 24], [21, 32], [27, 40]],
            np.int64,
        ),
    ]
    for actual, expected, dtype in cases:
        assert actual.dtype == dtype
        assert_array_equal(actual, expected)
    assert sl.block_diag().shape == (1, 0)
    assert sl.leslie([[1.0, 2.0], [3.0, 4.0]], [[0.5], [0.25]]).shape == (2, 2, 2)
    assert sl.khatri_rao(np.ones((2, 3)), np.ones((4, 3))).shape == (8, 3)


def test_dft_matrix_keeps_signed_zeros():
    m = sl.dft(3)
    assert m.dtype == np.complex128
    assert_array_equal(
        m,
        [
            [1, 1, 1],
            [1, -0.4999999999999998 - 0.8660254037844387j, -0.5000000000000003 + 0.8660254037844384j],
            [1, -0.5000000000000004 + 0.8660254037844384j, -0.4999999999999991 - 0.8660254037844392j],
        ],
    )
    assert_array_equal(np.signbit(m[0].imag), [False, True, True])
    assert_array_equal(
        sl.dft(2, scale="sqrtn"),
        [[0.7071067811865475, 0.7071067811865475], [0.7071067811865475, -0.7071067811865475 - 8.659560562354932e-17j]],
    )


def test_blas_and_lapack_wrappers():
    nrm2 = blas.get_blas_funcs("nrm2", (b,))
    assert (nrm2.__name__, nrm2.typecode, nrm2.prefix, nrm2.module_name) == ("function dnrm2", "d", "d", "fblas")
    assert repr(nrm2) == "<fortran function dnrm2>"
    assert nrm2(b) == 3.7416573867739413
    getrf, getrs = lapack.get_lapack_funcs(("getrf", "getrs"), (G,))
    assert getrf.__name__ == "function dgetrf" and getrf.dtype == np.float64
    lu, piv, info = getrf(G)
    assert info == 0 and piv.dtype == np.int32
    assert_array_equal(piv, [2, 2, 2])
    assert_array_equal(lu, sl.lu_factor(G)[0])
    x, info = getrs(lu, piv, b)
    assert info == 0
    assert_array_equal(x, [-0.3333333333333333, 0.6666666666666666, -0.0])
    assert blas.find_best_blas_type((np.float32(1),)) == ("s", np.dtype("float32"), True)
    assert lapack.find_best_lapack_type((np.ones(2, dtype=np.float32), np.ones(2))) == ("d", np.dtype("float64"), True)
    with pytest.raises(Exception, match=r"\(trans>=0 && trans <=2\) failed for 1st keyword trans: dgetrs:trans=3") as raised:
        sl.lu_solve(sl.lu_factor(np.eye(2)), np.ones(2), trans=3)
    assert (type(raised.value).__module__, type(raised.value).__name__) == ("_flapack", "error")
    with pytest.raises(Exception, match="failed for 2nd keyword trans: dtrtrs:trans=-1"):
        lapack.dtrtrs(np.eye(2), np.ones(2), trans=-1)


def test_lapack_outputs_keep_fortran_order():
    columns = np.ones((3, 2))
    fortran = [
        sl.lu_factor(G)[0],
        sl.lu_solve(sl.lu_factor(G), columns),
        sl.cho_solve(sl.cho_factor(S), columns),
        sl.solve_triangular(np.tril(G), columns, lower=True),
        sl.solve_banded((1, 1), np.ones((3, 3)), columns),
        sl.qr(G, mode="raw")[0][0],
        sl.eigh(S)[1],
        lapack.dgetrf(G)[0],
        lapack.dpotrf(S)[0],
        lapack.dtrtrs(np.triu(G), columns)[0],
    ]
    for array in fortran:
        assert array.flags.f_contiguous and not array.flags.c_contiguous
    c_order = [sl.solve(G, columns), sl.inv(G), sl.lu(G)[1], sl.cholesky(S), sl.qr(G)[0], sl.svd(G)[0], sl.expm(G / 10)]
    for array in c_order:
        assert array.flags.c_contiguous and not array.flags.f_contiguous


def test_singular_and_ill_conditioned_input():
    _, caught = recorded(sl.solve, np.array([[1.0, 1.0], [1.0, 1.0 + 2.0**-52]]), np.array([1.0, 2.0]))
    assert caught == [("LinAlgWarning", "An ill-conditioned matrix detected: slice 0 has rcond = 5.551115123125783e-17.")]
    _, caught = recorded(sl.lu_factor, np.array([[1.0, 2.0], [2.0, 4.0]]))
    assert caught == [("LinAlgWarning", "Diagonal number 2 is exactly zero. Singular matrix.")]
    singular = "A singular matrix detected: slice\\(s\\) \\[0\\] are singular."
    with pytest.raises(sl.LinAlgError, match=singular):
        sl.solve(np.array([[1.0, 2.0], [2.0, 4.0]]), np.array([1.0, 2.0]))
    with pytest.raises(sl.LinAlgError, match=singular):
        sl.inv(np.array([[1.0, 2.0], [2.0, 4.0]]))
    with pytest.raises(sl.LinAlgError, match="Internal potrf return info = \\[2\\] for slices \\[0\\]."):
        sl.cholesky(np.array([[1.0, 2.0], [2.0, 1.0]]))
    with pytest.raises(sl.LinAlgError, match="singular matrix: resolution failed at diagonal 1"):
        sl.solve_triangular(np.array([[1.0, 0.0], [1.0, 0.0]]), np.ones(2), lower=True)
    assert sl.LinAlgError is np.linalg.LinAlgError
    assert issubclass(sl.LinAlgWarning, RuntimeWarning)


# name: (call, exception type, message)
INVALID_INPUT = {
    "solve_nonsquare": (lambda: sl.solve(np.ones((2, 3)), np.ones(2)), ValueError, "Expected square matrix, got a1.shape=(2, 3)"),
    "inv_nonsquare": (lambda: sl.inv(np.ones((2, 3))), ValueError, "Expected square matrix, got a1.shape=(2, 3)"),
    "det_nonsquare": (
        lambda: sl.det(np.ones((2, 3))),
        ValueError,
        "Last 2 dimensions of the array must be square but received shape (2, 3).",
    ),
    "solve_shapes": (lambda: sl.solve(np.eye(3), np.ones(2)), ValueError, "incompatible shapes: a1.shape=(3, 3) and b1.shape=(2, 1)"),
    "solve_structure": (lambda: sl.solve(np.eye(2), np.ones(2), assume_a="bogus"), ValueError, "bogus is not a recognized matrix structure"),
    "cholesky_nonsquare": (
        lambda: sl.cholesky(np.ones((2, 3))),
        ValueError,
        "Expected a square matrix or batch thereof, got a1.shape=(2, 3)",
    ),
    "inv_nan": (
        lambda: sl.inv(np.array([[np.nan, 1.0], [1.0, 1.0]])),
        ValueError,
        "array must not contain infs or NaNs",
    ),
    "solve_inf": (
        lambda: sl.solve(np.array([[np.inf, 1.0], [1.0, 1.0]]), np.ones(2)),
        ValueError,
        "array must not contain infs or NaNs",
    ),
    "banded_shape": (
        lambda: sl.solve_banded((1, 1), np.ones((2, 3)), np.ones(3)),
        ValueError,
        "invalid values for the number of lower and upper diagonals: l+u+1 (3) does not equal ab.shape[0] (2)",
    ),
    "eigh_nonsquare": (lambda: sl.eigh(np.ones((2, 3))), ValueError, 'expected square "a" matrix'),
    "eigh_subset": (
        lambda: sl.eigh(np.eye(3), subset_by_index=[2, 5]),
        ValueError,
        "Requested eigenvalue indices are not valid. Valid range is [0, 2] and start <= end, but "
        "start=2, end=5 is given",
    ),
    "lstsq_shapes": (
        lambda: sl.lstsq(np.ones((3, 2)), np.ones(2)),
        ValueError,
        "Shape mismatch: a and b should have the same number of rows (3 != 2).",
    ),
    "lstsq_driver": (lambda: sl.lstsq(np.ones((3, 2)), np.ones(3), lapack_driver="bogus"), ValueError, 'LAPACK driver "bogus" is not found'),
    "expm_nonsquare": (lambda: sl.expm(np.ones((2, 3))), sl.LinAlgError, "Last 2 dimensions of the array must be square"),
    "norm_order": (lambda: sl.norm(G, 3), ValueError, "Invalid norm order for matrices."),
    "qr_mode": (
        lambda: sl.qr(G, mode="bogus"),
        ValueError,
        "Mode argument should be one of ['full', 'qr', 'r', 'raw', 'economic']",
    ),
    "hadamard_order": (lambda: sl.hadamard(3), ValueError, "n must be a positive integer, and n must be a power of 2"),
    "svd_driver": (lambda: sl.svd(G, lapack_driver="bogus"), ValueError, 'lapack_driver must be "gesdd" or "gesvd", not "bogus"'),
    "companion_leading_zero": (
        lambda: sl.companion([0.0, 1.0, 2.0]),
        ValueError,
        "The first coefficient(s) of `a` (i.e. elements of `a[..., 0]`) must not be zero.",
    ),
    "issymmetric_nonsquare": (lambda: sl.issymmetric(np.ones((2, 3))), ValueError, "Input array must be square."),
}


@pytest.mark.parametrize(
    "case",
    [
        "solve_nonsquare",
        "inv_nonsquare",
        "det_nonsquare",
        "solve_shapes",
        "solve_structure",
        "cholesky_nonsquare",
        "inv_nan",
        "solve_inf",
        "banded_shape",
        "eigh_nonsquare",
        "eigh_subset",
        "lstsq_shapes",
        "lstsq_driver",
        "expm_nonsquare",
        "norm_order",
        "qr_mode",
        "hadamard_order",
        "svd_driver",
        "companion_leading_zero",
        "issymmetric_nonsquare",
    ],
)
def test_invalid_input_raises_scipy_errors(case):
    call, error, message = INVALID_INPUT[case]
    with pytest.raises(error) as raised:
        call()
    assert str(raised.value) == message
