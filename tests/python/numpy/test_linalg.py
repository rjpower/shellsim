# Portable NumPy semantics. Expectations checked against NumPy 2.5.3 on CPython 3.14.4.
# Scope: dot products, matmul at every rank, and numpy.linalg decompositions checked by invariants.

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal


def spd_matrix():
    return np.array([[4.0, 2.0, 0.6], [2.0, 5.0, 1.0], [0.6, 1.0, 3.0]])


def general_matrix():
    return np.array([[2.0, -1.0, 0.0], [1.0, 3.0, 2.0], [0.0, 1.0, 4.0]])


def test_dot_of_vectors_and_matrices():
    a = np.array([1, 2, 3])
    b = np.array([4, 5, 6])
    assert np.dot(a, b) == 32
    assert np.dot(a, b).dtype == np.int64
    m = np.array([[1, 2], [3, 4]])
    assert_array_equal(np.dot(m, m), [[7, 10], [15, 22]])
    assert_array_equal(np.dot(m, np.array([1, 1])), [3, 7])
    assert_array_equal(m.dot(np.array([1, 0])), [1, 3])


def test_dot_with_scalar_scales():
    assert_array_equal(np.dot(3, np.array([1, 2])), [3, 6])


def test_vdot_flattens_and_conjugates_first_argument():
    assert np.vdot(np.array([[1, 2], [3, 4]]), np.array([[1, 1], [1, 1]])) == 10
    a = np.array([1 + 2j, 3 - 1j])
    b = np.array([2 + 0j, 1j])
    assert np.vdot(a, b) == (2 - 4j) + (-1 + 3j)
    assert np.vdot(a, b) == np.dot(np.conj(a), b)


def test_inner_and_outer():
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([0.5, -1.0, 2.0])
    assert np.inner(a, b) == 4.5
    assert_array_equal(np.outer(np.array([1, 2]), np.array([3, 4, 5])), [[3, 4, 5], [6, 8, 10]])
    m = np.array([[1, 2], [3, 4]])
    assert_array_equal(np.inner(m, m), [[5, 11], [11, 25]])


def test_matmul_vector_vector_is_scalar():
    result = np.array([1.0, 2.0]) @ np.array([3.0, 4.0])
    assert result == 11.0
    assert np.ndim(result) == 0


def test_matmul_matrix_vector():
    m = np.array([[1, 2, 3], [4, 5, 6]])
    v = np.array([1, 0, -1])
    result = m @ v
    assert result.shape == (2,)
    assert_array_equal(result, [-2, -2])


def test_matmul_vector_matrix():
    m = np.array([[1, 2, 3], [4, 5, 6]])
    v = np.array([1, -1])
    result = v @ m
    assert result.shape == (3,)
    assert_array_equal(result, [-3, -3, -3])


def test_matmul_matrix_matrix():
    a = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    b = np.array([[1.0, 0.0, 2.0], [0.0, 1.0, -1.0]])
    result = np.matmul(a, b)
    assert result.shape == (3, 3)
    assert_array_equal(result, [[1.0, 2.0, 0.0], [3.0, 4.0, 2.0], [5.0, 6.0, 4.0]])


def test_matmul_batched_broadcasts_leading_dimensions():
    stack = np.arange(12).reshape(3, 2, 2)
    single = np.array([[1, 1], [0, 1]])
    result = stack @ single
    assert result.shape == (3, 2, 2)
    for index in range(3):
        assert_array_equal(result[index], stack[index] @ single)
    assert_array_equal(result[1], [[4, 9], [6, 13]])


def test_matmul_batched_with_vector():
    stack = np.arange(8).reshape(2, 2, 2)
    result = stack @ np.array([1, 2])
    assert result.shape == (2, 2)
    assert_array_equal(result, [[2, 8], [14, 20]])


def test_matmul_shape_mismatch_raises_value_error():
    with pytest.raises(ValueError):
        np.ones((2, 3)) @ np.ones((2, 3))
    with pytest.raises(ValueError):
        np.matmul(np.ones(3), np.ones(4))


def test_dot_shape_mismatch_raises_value_error():
    with pytest.raises(ValueError):
        np.dot(np.ones((2, 3)), np.ones((2, 3)))


def test_matmul_rejects_scalars():
    with pytest.raises(ValueError):
        np.matmul(np.array(2.0), np.ones(2))


def test_norm_vector_orders():
    v = np.array([3.0, -4.0])
    assert np.linalg.norm(v) == 5.0
    assert np.linalg.norm(v, ord=1) == 7.0
    assert np.linalg.norm(v, ord=np.inf) == 4.0
    assert np.linalg.norm(v, ord=-np.inf) == 3.0


def test_norm_matrix_frobenius_and_axis():
    m = np.array([[1.0, 2.0], [3.0, 4.0]])
    assert_allclose(np.linalg.norm(m), np.sqrt(30.0), rtol=1e-15)
    assert_allclose(np.linalg.norm(m, "fro"), np.sqrt(30.0), rtol=1e-15)
    assert_allclose(np.linalg.norm(m, axis=0), [np.sqrt(10.0), np.sqrt(20.0)], rtol=1e-15)
    assert_allclose(np.linalg.norm(m, axis=1), [np.sqrt(5.0), 5.0], rtol=1e-15)
    assert_array_equal(np.linalg.norm(m, ord=1, axis=1), [3.0, 7.0])


def test_norm_of_int_vector_is_float():
    result = np.linalg.norm(np.array([3, 4]))
    assert result == 5.0
    assert result.dtype == np.float64


def test_inv_times_matrix_is_identity():
    a = general_matrix()
    inverse = np.linalg.inv(a)
    assert_allclose(a @ inverse, np.eye(3), atol=1e-12)
    assert_allclose(inverse @ a, np.eye(3), atol=1e-12)


def test_inv_of_simple_matrix():
    assert_allclose(np.linalg.inv(np.array([[4.0, 7.0], [2.0, 6.0]])), [[0.6, -0.7], [-0.2, 0.4]], atol=1e-15)


def test_inv_singular_raises_linalg_error():
    with pytest.raises(np.linalg.LinAlgError):
        np.linalg.inv(np.array([[1.0, 2.0], [2.0, 4.0]]))


def test_linalg_error_is_value_error():
    assert issubclass(np.linalg.LinAlgError, ValueError)


def test_inv_non_square_raises_linalg_error():
    with pytest.raises(np.linalg.LinAlgError):
        np.linalg.inv(np.ones((2, 3)))


def test_solve_vector_and_matrix_right_hand_sides():
    a = np.array([[3.0, 1.0], [1.0, 2.0]])
    b = np.array([9.0, 8.0])
    x = np.linalg.solve(a, b)
    assert_allclose(x, [2.0, 3.0], atol=1e-14)
    rhs = np.array([[9.0, 1.0], [8.0, 2.0]])
    xs = np.linalg.solve(a, rhs)
    assert xs.shape == (2, 2)
    assert_allclose(a @ xs, rhs, atol=1e-14)


def test_solve_singular_raises_linalg_error():
    with pytest.raises(np.linalg.LinAlgError):
        np.linalg.solve(np.array([[1.0, 2.0], [2.0, 4.0]]), np.array([1.0, 2.0]))


def test_det():
    assert_allclose(np.linalg.det(np.array([[1.0, 2.0], [3.0, 4.0]])), -2.0, rtol=1e-14)
    assert_allclose(np.linalg.det(general_matrix()), 24.0, rtol=1e-14)
    assert_allclose(np.linalg.det(np.eye(4)), 1.0, rtol=1e-15)
    assert np.linalg.det(np.array([[1.0, 2.0], [2.0, 4.0]])) == 0.0


def test_det_batched():
    stack = np.array([[[2.0, 0.0], [0.0, 3.0]], [[1.0, 2.0], [3.0, 4.0]]])
    assert_allclose(np.linalg.det(stack), [6.0, -2.0], rtol=1e-14)


def test_slogdet():
    sign, logdet = np.linalg.slogdet(np.array([[1.0, 2.0], [3.0, 4.0]]))
    assert sign == -1.0
    assert_allclose(logdet, np.log(2.0), rtol=1e-14)
    sign, logdet = np.linalg.slogdet(general_matrix())
    assert sign == 1.0
    assert_allclose(logdet, np.log(24.0), rtol=1e-14)


def test_slogdet_of_singular_matrix():
    sign, logdet = np.linalg.slogdet(np.array([[1.0, 2.0], [2.0, 4.0]]))
    assert sign == 0.0
    assert logdet == -np.inf


@pytest.mark.parametrize(
    "rows, expected",
    [
        ([[1, 0], [0, 1]], 2),
        ([[1, 2], [2, 4]], 1),
        ([[0, 0], [0, 0]], 0),
        ([[1, 2, 3], [4, 5, 6], [7, 8, 9]], 2),
        ([[1, 2, 3], [4, 5, 6]], 2),
    ],
)
def test_matrix_rank(rows, expected):
    assert np.linalg.matrix_rank(np.array(rows, dtype=float)) == expected


def test_matrix_power():
    m = np.array([[1, 1], [1, 0]])
    assert_array_equal(np.linalg.matrix_power(m, 0), np.eye(2, dtype=int))
    assert_array_equal(np.linalg.matrix_power(m, 1), m)
    assert_array_equal(np.linalg.matrix_power(m, 10), [[89, 55], [55, 34]])
    assert np.linalg.matrix_power(m, 10).dtype == np.int64


def test_matrix_power_negative_uses_inverse():
    m = np.array([[2.0, 0.0], [0.0, 4.0]])
    assert_allclose(np.linalg.matrix_power(m, -2), [[0.25, 0.0], [0.0, 0.0625]], rtol=1e-15)


def test_lstsq_overdetermined_line_fit():
    x = np.array([0.0, 1.0, 2.0, 3.0])
    y = np.array([-1.0, 0.2, 0.9, 2.1])
    a = np.vstack([x, np.ones(len(x))]).T
    solution, residuals, rank, singular_values = np.linalg.lstsq(a, y, rcond=None)
    assert_allclose(solution, [1.0, -0.95], atol=1e-12)
    assert residuals.shape == (1,)
    assert_allclose(residuals, [0.05], atol=1e-12)
    assert rank == 2
    assert singular_values.shape == (2,)
    assert singular_values[0] >= singular_values[1]


def test_lstsq_exact_square_system_has_empty_residuals():
    a = np.array([[3.0, 1.0], [1.0, 2.0]])
    solution, residuals, rank, singular_values = np.linalg.lstsq(a, np.array([9.0, 8.0]), rcond=None)
    assert_allclose(solution, [2.0, 3.0], atol=1e-12)
    assert residuals.shape == (0,)
    assert rank == 2


def test_pinv_satisfies_moore_penrose_conditions():
    a = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    p = np.linalg.pinv(a)
    assert p.shape == (2, 3)
    assert_allclose(a @ p @ a, a, atol=1e-12)
    assert_allclose(p @ a @ p, p, atol=1e-12)
    assert_allclose(p @ a, np.eye(2), atol=1e-12)


def test_pinv_of_invertible_matrix_is_inverse():
    a = general_matrix()
    assert_allclose(np.linalg.pinv(a), np.linalg.inv(a), atol=1e-12)


def test_qr_reconstructs_with_orthonormal_q():
    a = np.array([[12.0, -51.0, 4.0], [6.0, 167.0, -68.0], [-4.0, 24.0, -41.0]])
    q, r = np.linalg.qr(a)
    assert q.shape == (3, 3)
    assert r.shape == (3, 3)
    assert_allclose(q.T @ q, np.eye(3), atol=1e-12)
    assert_allclose(q @ r, a, atol=1e-12)
    assert_array_equal(np.tril(r, -1), np.zeros((3, 3)))


def test_qr_reduced_shapes_for_tall_matrix():
    a = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
    q, r = np.linalg.qr(a)
    assert q.shape == (4, 2)
    assert r.shape == (2, 2)
    assert_allclose(q.T @ q, np.eye(2), atol=1e-12)
    assert_allclose(q @ r, a, atol=1e-12)
    assert r[1, 0] == 0.0


def test_cholesky_is_lower_triangular_factor():
    a = spd_matrix()
    lower = np.linalg.cholesky(a)
    assert_array_equal(np.triu(lower, 1), np.zeros((3, 3)))
    assert np.all(np.diag(lower) > 0)
    assert_allclose(lower @ lower.T, a, atol=1e-12)
    assert_allclose(lower[0, 0], 2.0, rtol=1e-15)


def test_cholesky_rejects_non_positive_definite():
    with pytest.raises(np.linalg.LinAlgError):
        np.linalg.cholesky(np.array([[1.0, 2.0], [2.0, 1.0]]))


def test_eigh_eigenvalues_ascending_and_eigenvectors():
    a = np.array([[2.0, 1.0], [1.0, 2.0]])
    w, v = np.linalg.eigh(a)
    assert_allclose(w, [1.0, 3.0], atol=1e-14)
    assert_allclose(a @ v, v * w, atol=1e-12)
    assert_allclose(v.T @ v, np.eye(2), atol=1e-12)


def test_eigh_three_by_three():
    a = spd_matrix()
    w, v = np.linalg.eigh(a)
    assert w.shape == (3,)
    assert v.shape == (3, 3)
    assert w[0] <= w[1] <= w[2]
    assert_allclose(np.sum(w), np.trace(a), rtol=1e-13)
    assert_allclose(np.prod(w), np.linalg.det(a), rtol=1e-12)
    assert_allclose(a @ v, v * w, atol=1e-12)
    assert_allclose(v.T @ v, np.eye(3), atol=1e-12)


def test_eigh_diagonal_matrix_sorts_eigenvalues():
    w, v = np.linalg.eigh(np.diag([3.0, -1.0, 2.0]))
    assert_allclose(w, [-1.0, 2.0, 3.0], atol=1e-15)
    assert_allclose(np.abs(v), [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], atol=1e-12)


def test_svd_reconstructs_with_descending_singular_values():
    a = np.array([[3.0, 2.0, 2.0], [2.0, 3.0, -2.0]])
    u, s, vt = np.linalg.svd(a)
    assert u.shape == (2, 2)
    assert s.shape == (2,)
    assert vt.shape == (3, 3)
    assert_allclose(s, [5.0, 3.0], atol=1e-12)
    assert_allclose(u @ np.diag(s) @ vt[:2], a, atol=1e-12)
    assert_allclose(u.T @ u, np.eye(2), atol=1e-12)
    assert_allclose(vt @ vt.T, np.eye(3), atol=1e-12)


def test_svd_reduced_shapes():
    a = np.arange(12.0).reshape(4, 3)
    u, s, vt = np.linalg.svd(a, full_matrices=False)
    assert u.shape == (4, 3)
    assert s.shape == (3,)
    assert vt.shape == (3, 3)
    assert s[0] >= s[1] >= s[2] >= 0.0
    assert_allclose(u @ np.diag(s) @ vt, a, atol=1e-12)
    assert_allclose(u.T @ u, np.eye(3), atol=1e-12)


def test_svd_values_only():
    s = np.linalg.svd(np.array([[0.0, 2.0], [1.0, 0.0]]), compute_uv=False)
    assert_allclose(s, [2.0, 1.0], atol=1e-15)


def test_decompositions_return_float64_for_int_input():
    a = np.array([[2, 1], [1, 2]])
    assert np.linalg.inv(a).dtype == np.float64
    assert np.linalg.det(a).dtype == np.float64
    assert np.linalg.eigh(a)[0].dtype == np.float64
    assert np.linalg.svd(a)[1].dtype == np.float64
