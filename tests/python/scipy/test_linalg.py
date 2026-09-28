# Portable SciPy semantics. Expectations checked against SciPy 1.18.1 and NumPy 2.5.3 on
# CPython 3.14.4.
# Scope: shellsim's `scipy.linalg` subset (see `source/scipy/linalg.py`'s module docstring for
# what is kept and what is dropped -- there is no `scipy.linalg.lapack`/`.blas`).
# shellsim's dense kernels are original, backward-stable implementations, not a port of LAPACK or
# OpenBLAS (see docs/scipy.md), so exact bits rarely match SciPy's OpenBLAS build. Deterministic
# algorithms (partial-pivoted LU, Cholesky, Householder QR, banded/triangular elimination) agree
# with SciPy to about 1e-14; iterative ones (`eig`, `eigh`, `svd`, `lstsq`, `pinv`) agree to about
# 1e-13 relative and choose eigenvector/singular-vector signs independently, so those are checked
# by their defining invariants (e.g. `A @ v == w * v`) rather than by literal fixtures.

import warnings

import numpy as np
import pytest
import scipy.linalg as sl
from numpy.testing import assert_allclose, assert_array_equal

G = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 10.0]])
S = np.array([[4.0, 1.0, 2.0], [1.0, 5.0, 3.0], [2.0, 3.0, 6.0]])
b = np.array([1.0, 2.0, 3.0])
R = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])


def close(actual, expected, rtol=1e-13, atol=1e-14):
    assert_allclose(actual, expected, rtol=rtol, atol=atol)


def recorded(function, *args, **kwargs):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = function(*args, **kwargs)
    return result, [warning.category for warning in caught]


def test_solve_general_symmetric_and_positive_definite():
    close(sl.solve(G, b), [-0.3333333333333333, 0.6666666666666666, -0.0])
    close(sl.solve(G, b, transposed=True), [1.0, -0.0, 0.0])
    positive = [-2.0816681711721685e-17, 0.14285714285714277, 0.42857142857142866]
    close(sl.solve(S, b, assume_a="pos"), positive)
    close(sl.solve(S, b, assume_a="sym"), [0.0, 0.14285714285714285, 0.4285714285714286])
    close(sl.solve(S, b, assume_a="gen"), positive, rtol=1e-9)
    close(
        sl.solve(np.tril(S), b, assume_a="sym", lower=True),
        [0.0, 0.14285714285714285, 0.42857142857142855],
    )
    rhs = np.array([[9.0, 1.0], [8.0, 2.0], [1.0, 0.0]])
    xs = sl.solve(G, rhs)
    assert xs.shape == (3, 2)
    close(G @ xs, rhs)


def test_solve_batches_over_a_stack_of_matrices():
    # SciPy's own `solve` requires `b.ndim == a.ndim` for a batch (a lone leading vector-batch
    # shape is ambiguous and SciPy rejects it), so the right-hand side is a stack of columns.
    result = sl.solve(np.stack([G, S]), np.stack([b, b])[..., None])
    assert result.shape == (2, 3, 1)
    close(G @ result[0], b[:, None])
    close(S @ result[1], b[:, None])


def test_solve_promotes_and_preserves_dtype():
    result = sl.solve(np.array([[2, 1], [1, 3]]), np.array([1, 2]))
    assert result.dtype == np.float64
    close(result, [0.19999999999999998, 0.6])
    single = sl.solve(G.astype(np.float32), b.astype(np.float32))
    assert single.dtype == np.float32
    close(single, [-0.3333333, 0.6666667, 0.0], rtol=1e-6, atol=1e-6)
    # SciPy also warns that float16 input is deprecated; shellsim just promotes it.
    result, _ = recorded(
        sl.solve, np.array([[2.0, 1.0], [1.0, 3.0]], dtype=np.float16), np.array([1.0, 2.0], dtype=np.float16)
    )
    assert result.dtype == np.float32


def test_solve_triangular_and_banded():
    close(sl.solve_triangular(np.tril(G), b, lower=True), [1.0, -0.4, -0.07999999999999999])
    close(sl.solve_triangular(np.triu(G), b, trans="T"), [1.0, 0.0, 0.0])
    close(sl.solve_triangular(np.tril(G), b, lower=True, unit_diagonal=True), [1.0, -2.0, 12.0])
    t = np.tril(np.array([[3.0, 0, 0], [1, 2, 0], [4, 5, 6]]))
    x = sl.solve_triangular(t, b, lower=True)
    close(t @ x, b)
    bands = np.array([[0.0, 1.0, 2.0], [4.0, 5.0, 6.0], [7.0, 8.0, 0.0]])
    close(sl.solve_banded((1, 1), bands, b), [0.5714285714285718, -1.285714285714287, 2.2142857142857157])
    close(
        sl.solve_banded((1, 1), bands, np.stack([b, 2 * b], axis=1)),
        [[0.5714285714285718, 1.1428571428571437], [-1.285714285714287, -2.571428571428574], [2.2142857142857157, 4.4285714285714315]],
    )
    wide = np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 2.0], [4.0, 5.0, 6.0], [7.0, 8.0, 0.0]])
    close(sl.solve_banded((1, 2), wide, b), [0.12857142857142861, 0.042857142857142816, 0.4428571428571429])
    close(sl.solve_banded((0, 1), np.array([[0.0, 1.0, 2.0], [4.0, 5.0, 6.0]]), b), [0.2, 0.2, 0.5])


def test_inv_and_det():
    close(
        sl.inv(G),
        [
            [-0.6666666666666662, -1.333333333333333, 0.9999999999999996],
            [-0.6666666666666676, 3.6666666666666665, -1.9999999999999996],
            [1.0000000000000004, -2.0, 0.9999999999999999],
        ],
    )
    close(
        sl.inv(S, assume_a="pos"),
        [
            [0.3, 1.3877787807814457e-17, -0.10000000000000002],
            [1.3877787807814457e-17, 0.28571428571428564, -0.14285714285714285],
            [-0.10000000000000002, -0.14285714285714285, 0.27142857142857146],
        ],
    )
    det = sl.det(G)
    assert type(det) is np.float64
    close(det, -3.0)
    close(sl.det(np.array([[2, 1], [1, 3]])), 5.0)
    close(sl.det(np.stack([G, S])), [-3.0, 70.0])
    close(sl.inv(np.stack([np.eye(2), 2 * np.eye(2)])), [np.eye(2), 0.5 * np.eye(2)])
    single = sl.inv(G.astype(np.float32))
    assert single.dtype == np.float32
    close(single, sl.inv(G).astype(np.float32), rtol=1e-5, atol=1e-5)


def test_lu_family():
    p, l, u = sl.lu(G)
    assert_array_equal(p, [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])
    assert_array_equal(l, [[1.0, 0.0, 0.0], [0.14285714285714285, 1.0, 0.0], [0.5714285714285714, 0.5000000000000002, 1.0]])
    close(u, [[7.0, 8.0, 10.0], [0.0, 0.8571428571428572, 1.5714285714285716], [0.0, 0.0, -0.5]])
    close(p @ l @ u, G)
    permuted, u2 = sl.lu(G, permute_l=True)
    close(permuted, p @ l)
    close(u2, u)
    indices, l2, u3 = sl.lu(G, p_indices=True)
    assert indices.dtype == np.int32
    assert_array_equal(indices, [1, 2, 0])
    close(G, (l2 @ u3)[indices])
    p2, l2, u2 = sl.lu(R)
    close(p2 @ l2 @ u2, R)
    p3, l3, u3 = sl.lu(np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))
    close(p3 @ l3 @ u3, [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    lu, piv = sl.lu_factor(G)
    assert piv.dtype == np.int32
    assert_array_equal(piv, [2, 2, 2])
    close(
        lu, [[7.0, 8.0, 10.0], [0.14285714285714285, 0.8571428571428572, 1.5714285714285716], [0.5714285714285714, 0.5000000000000002, -0.5]],
    )
    close(sl.lu_solve((lu, piv), b), [-0.3333333333333333, 0.6666666666666666, -0.0])
    close(sl.lu_solve((lu, piv), b, trans=1), [1.0, -0.0, 0.0])
    close(sl.lu_solve((lu, piv), b, trans=2), [1.0, -0.0, 0.0])
    close(sl.lu(np.stack([G, S]))[2][1], [[4.0, 1.0, 2.0], [0.0, 4.75, 2.5], [0.0, 0.0, 3.6842105263157894]])
    single, _ = sl.lu_factor(G.astype(np.float32))
    close(single, lu.astype(np.float32), rtol=1e-5, atol=1e-5)


def test_cholesky_family():
    upper = [[2.0, 0.5, 1.0], [0.0, 2.179449471770337, 1.1470786693528088], [0.0, 0.0, 1.9194297398747862]]
    close(sl.cholesky(S), upper)
    close(sl.cholesky(S, lower=True), np.array(upper).T)
    c, lower = sl.cho_factor(S)
    close(c, upper)
    assert isinstance(lower, np.ndarray) and lower.shape == () and not lower
    c_lower, lower = sl.cho_factor(S, lower=True)
    close(c_lower, np.array(upper).T)
    expected = [-2.0816681711721685e-17, 0.14285714285714277, 0.42857142857142866]
    close(sl.cho_solve((c, False), b), expected)
    close(sl.cho_solve((c_lower, True), b), expected)
    close(sl.cholesky(np.stack([S, 2 * S + 3 * np.eye(3)]))[1], sl.cholesky(2 * S + 3 * np.eye(3)))


def test_qr_modes():
    q, r = sl.qr(G)
    close(r, [[-8.124038404635959, -9.601136296387955, -11.939874624995277], [0.0, 0.9045340337332926, 1.50755672288882], [0.0, 0.0, 0.40824829046386224]])
    close(q @ q.T, np.eye(3))
    close(q @ r, G)
    (r_only,) = sl.qr(G, mode="r")
    close(r_only, r)
    q2, r2 = sl.qr(R, mode="economic")
    assert q2.shape == (3, 2) and r2.shape == (2, 2)
    close(q2 @ r2, R)
    close(q2.T @ q2, np.eye(2))
    assert [x.shape for x in sl.qr(R)] == [(3, 3), (3, 2)]
    assert [x.shape for x in sl.qr(R, mode="economic")] == [(3, 2), (2, 2)]


def test_eig_and_eigvals_satisfy_the_eigenvector_equation():
    w, v = sl.eig(G)
    assert w.dtype == np.complex128
    close(G @ v, v * w, rtol=1e-9, atol=1e-9)
    close(np.sort(w.real), np.sort(sl.eigvals(G).real), rtol=1e-9)
    only_values = sl.eig(G, right=False)
    close(np.sort(only_values.real), np.sort(w.real), rtol=1e-9)
    rotation = np.array([[0.0, -1.0], [1.0, 0.0]])
    wr, vr = sl.eig(rotation)
    close(np.sort(wr.imag), [-1.0, 1.0], atol=1e-12)
    close(rotation @ vr, vr * wr, atol=1e-12)
    with pytest.raises(ValueError):
        sl.eig(np.ones((2, 3)))


def test_eigh_plain_generalized_and_subset():
    w, v = sl.eigh(S)
    close(w, [2.1943971674224088, 3.3867701566075468, 9.418832675970034])
    close(S @ v, v * w)
    close(v.T @ v, np.eye(3))
    generalized = [0.9183340005338673, 2.5000000000000004, 5.081665999466133]
    close(sl.eigvalsh(S, np.diag([1.0, 2.0, 3.0])), generalized, rtol=1e-9)
    close(sl.eigh(S, np.diag([1.0, 2.0, 3.0]))[0], generalized, rtol=1e-9)
    close(sl.eigh(S, eigvals_only=True, subset_by_index=[1, 2]), [3.3867701566075468, 9.418832675970037])
    close(sl.eigh(S, eigvals_only=True, subset_by_value=(3.0, 10.0)), [3.3867701566075485, 9.41883267597004])
    close(sl.eigvalsh(np.triu(S), lower=False), [2.1943971674224065, 3.38677015660755, 9.41883267597004])
    with pytest.raises(ValueError):
        sl.eigh(S, subset_by_index=[2, 5])


def test_svd_and_svdvals():
    u, s, vh = sl.svd(G)
    close(s, [17.412505166808593, 0.8751613501104364, 0.19686652111743])
    close(u @ np.diag(s) @ vh, G)
    close(u.T @ u, np.eye(3))
    close(vh @ vh.T, np.eye(3))
    close(sl.svdvals(G), s)
    assert [x.shape for x in sl.svd(R, full_matrices=False)] == [(3, 2), (2,), (2, 2)]
    assert [x.shape for x in sl.svd(R)] == [(3, 3), (2,), (2, 2)]
    close(sl.svd(R, compute_uv=False), [9.525518091565107, 0.5143005806586443])


def test_least_squares_and_pseudo_inverses():
    x, residues, rank, singular = sl.lstsq(np.array([[1.0, 1.0], [1.0, 2.0], [1.0, 3.0]]), np.array([1.0, 2.0, 2.0]))
    close(x, [0.6666666666666663, 0.5000000000000002])
    assert residues.shape == ()
    close(residues, 0.16666666666666677)
    assert type(rank) is np.int64 and rank == 2
    close(singular, [4.079143328941734, 0.6004912172131635])
    x, residues, rank, singular = sl.lstsq(np.array([[1.0, 2.0], [2.0, 4.0], [3.0, 6.0]]), b)
    close(x, [0.19999999999999996, 0.39999999999999997])
    assert np.isnan(residues) and rank == 1
    target = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    x, residues, rank, _ = sl.lstsq(R, target)
    assert residues.shape == (2,) and rank == 2
    close(residues, np.sum((R @ x - target) ** 2, axis=0))
    with pytest.raises(ValueError):
        sl.lstsq(np.ones((3, 2)), np.ones(2))
    close(
        sl.pinv(R),
        [[-1.3333333333333337, -0.33333333333333287, 0.6666666666666665], [1.0833333333333335, 0.3333333333333329, -0.4166666666666665]],
    )
    close(R @ sl.pinv(R) @ R, R)
    assert sl.pinv(np.array([[1.0, 2.0], [2.0, 4.0], [3.0, 6.0]]), return_rank=True)[1] == 1


def test_null_space_and_orth():
    ns = sl.null_space(np.array([[1.0, 1.0, 1.0]]))
    assert ns.shape == (3, 2)
    close(ns.T @ ns, np.eye(2))
    close(np.array([[1.0, 1.0, 1.0]]) @ ns, np.zeros((1, 2)), atol=1e-12)
    close(np.abs(sl.null_space(np.array([[1.0, 1.0]]))), [[0.7071067811865475], [0.7071067811865475]])
    close(np.abs(sl.orth(np.array([[1.0, 1.0], [1.0, 1.0]]))), [[0.7071067811865472], [0.7071067811865475]])
    assert sl.orth(R).shape == (3, 2)


def test_expm():
    close(
        sl.expm(np.array([[0.1, 0.2], [0.3, 1.9]])),
        [[1.1720433769251966, 0.6259875479854236], [0.9389813219781356, 6.805931308794008]],
    )
    assert_array_equal(sl.expm(np.zeros((2, 2))), np.eye(2))
    close(
        sl.expm(np.array([[10.0, 3.0], [2.0, -4.0]])),
        [[32459.13503668736, 6754.718933498614], [4503.145955665742, 937.1133470271582]],
        rtol=1e-9,
    )
    close(sl.expm(np.stack([np.zeros((2, 2)), np.eye(2)])), [np.eye(2), 2.718281828459045 * np.eye(2)])
    single = sl.expm(np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32))
    assert single.dtype == np.float32
    close(single, sl.expm(np.array([[0.1, 0.2], [0.3, 0.4]])).astype(np.float32), rtol=1e-5, atol=1e-5)
    with pytest.raises(sl.LinAlgError):
        sl.expm(np.ones((2, 3)))


def test_special_matrices():
    assert_array_equal(sl.toeplitz([1, 2, 3], [1, 4, 5]), [[1, 4, 5], [2, 1, 4], [3, 2, 1]])
    assert_array_equal(sl.toeplitz([1.0, 2.0, 3.0]), [[1.0, 2.0, 3.0], [2.0, 1.0, 2.0], [3.0, 2.0, 1.0]])
    assert_array_equal(sl.circulant([1, 2, 3]), [[1, 3, 2], [2, 1, 3], [3, 2, 1]])
    assert_array_equal(sl.block_diag([[1, 2]], [[3], [4]], 5), [[1, 2, 0, 0], [0, 0, 3, 0], [0, 0, 4, 0], [0, 0, 0, 5]])
    assert sl.block_diag().shape == (1, 0)
    close(sl.hilbert(3), [[1.0, 0.5, 0.3333333333333333], [0.5, 0.3333333333333333, 0.25], [0.3333333333333333, 0.25, 0.2]])


def test_linalg_error_and_warning_types():
    with pytest.raises(sl.LinAlgError):
        sl.solve(np.array([[1.0, 2.0], [2.0, 4.0]]), np.array([1.0, 2.0]))
    with pytest.raises(sl.LinAlgError):
        sl.inv(np.array([[1.0, 2.0], [2.0, 4.0]]))
    with pytest.raises(sl.LinAlgError):
        sl.cholesky(np.array([[1.0, 2.0], [2.0, 1.0]]))
    with pytest.raises(sl.LinAlgError):
        sl.solve_triangular(np.array([[1.0, 0.0], [1.0, 0.0]]), np.ones(2), lower=True)
    assert sl.LinAlgError is np.linalg.LinAlgError
    assert issubclass(sl.LinAlgWarning, RuntimeWarning)
    # `lu_factor` (and, through it, `lu`) warns on an exactly singular pivot; unlike SciPy's own
    # `gecon`-based estimate, shellsim does not warn `solve`/`inv` for a merely ill-conditioned
    # (but not exactly singular) matrix -- see `LinAlgWarning`'s docstring.
    _, caught = recorded(sl.lu_factor, np.array([[1.0, 2.0], [2.0, 4.0]]))
    assert caught == [sl.LinAlgWarning]


# name: (call, exception type)
INVALID_INPUT = {
    "solve_nonsquare": (lambda: sl.solve(np.ones((2, 3)), np.ones(2)), ValueError),
    "inv_nonsquare": (lambda: sl.inv(np.ones((2, 3))), ValueError),
    "det_nonsquare": (lambda: sl.det(np.ones((2, 3))), ValueError),
    "solve_shapes": (lambda: sl.solve(np.eye(3), np.ones(2)), ValueError),
    "solve_structure": (lambda: sl.solve(np.eye(2), np.ones(2), assume_a="bogus"), ValueError),
    "cholesky_nonsquare": (lambda: sl.cholesky(np.ones((2, 3))), ValueError),
    "inv_nan": (lambda: sl.inv(np.array([[np.nan, 1.0], [1.0, 1.0]])), ValueError),
    "solve_inf": (lambda: sl.solve(np.array([[np.inf, 1.0], [1.0, 1.0]]), np.ones(2)), ValueError),
    "banded_shape": (lambda: sl.solve_banded((1, 1), np.ones((2, 3)), np.ones(3)), ValueError),
    "eigh_nonsquare": (lambda: sl.eigh(np.ones((2, 3))), ValueError),
    "eigh_subset": (lambda: sl.eigh(np.eye(3), subset_by_index=[2, 5]), ValueError),
    "eig_nonsquare": (lambda: sl.eig(np.ones((2, 3))), ValueError),
    "lstsq_shapes": (lambda: sl.lstsq(np.ones((3, 2)), np.ones(2)), ValueError),
    "expm_nonsquare": (lambda: sl.expm(np.ones((2, 3))), sl.LinAlgError),
    "qr_mode": (lambda: sl.qr(G, mode="bogus"), ValueError),
}


@pytest.mark.parametrize("case", list(INVALID_INPUT))
def test_invalid_input_raises_scipy_errors(case):
    call, error = INVALID_INPUT[case]
    with pytest.raises(error):
        call()
