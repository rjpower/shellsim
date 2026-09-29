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


def test_complex_norm_uses_both_components():
    vector = np.array([3 + 4j, -5j])
    assert_allclose(np.linalg.norm(vector), np.sqrt(50.0))
    assert_allclose(np.linalg.norm(vector, ord=1), 10.0)
    matrix = np.array([[3 + 4j, 0], [0, 1j]], dtype=np.complex64)
    result = np.linalg.norm(matrix)
    assert result.dtype == np.float32
    assert_allclose(result, np.sqrt(26.0), rtol=1e-6)
    assert_allclose(np.linalg.norm(matrix, axis=0), [5.0, 1.0])
    assert_allclose(np.linalg.norm(matrix, ord=2), 5.0, atol=1e-12)
    assert_allclose(np.linalg.norm(matrix, ord="nuc"), 6.0, atol=1e-12)


def test_inv_times_matrix_is_identity():
    a = general_matrix()
    inverse = np.linalg.inv(a)
    assert_allclose(a @ inverse, np.eye(3), atol=1e-12)
    assert_allclose(inverse @ a, np.eye(3), atol=1e-12)


def test_inv_of_simple_matrix():
    assert_allclose(np.linalg.inv(np.array([[4.0, 7.0], [2.0, 6.0]])), [[0.6, -0.7], [-0.2, 0.4]], atol=1e-15)


def test_complex_inverse_and_solve():
    a = np.array([[2 + 1j, 1 - 2j], [1j, 3 - 1j]])
    inverse = np.linalg.inv(a)
    assert_allclose(a @ inverse, np.eye(2), atol=1e-12)
    b = np.array([1 + 2j, 3 - 1j])
    assert_allclose(a @ np.linalg.solve(a, b), b, atol=1e-12)
    assert_allclose(a @ np.linalg.solve(a, np.eye(2, dtype=complex)), np.eye(2), atol=1e-12)
    stacked = np.array([a, 2 * a])
    assert_allclose(stacked @ np.linalg.inv(stacked), np.broadcast_to(np.eye(2), (2, 2, 2)), atol=1e-12)
    b_stacked = np.array([b, 2 * b])
    result = np.linalg.solve(stacked, b_stacked[..., np.newaxis])
    assert_allclose(stacked @ result, b_stacked[..., np.newaxis], atol=1e-12)
    single = np.linalg.solve(a.astype(np.complex64), b.astype(np.complex64))
    assert single.dtype == np.complex64


def test_complex_linalg_rejects_vectors_where_matrices_are_required():
    vector = np.array([1 + 1j, 2j])
    for operation in (np.linalg.inv, np.linalg.det, np.linalg.cholesky, np.linalg.eigh, np.linalg.eig, np.linalg.svd, np.linalg.qr):
        with pytest.raises(np.linalg.LinAlgError):
            operation(vector)


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


def test_solve_batched_matrix_uses_matrix_core_dimensions():
    a = np.array([[[2.0, 0.0], [0.0, 3.0]], [[4.0, 0.0], [0.0, 5.0]]])
    rhs = np.eye(2)
    result = np.linalg.solve(a, rhs)
    assert result.shape == (2, 2, 2)
    assert_allclose(a @ result, np.broadcast_to(rhs, (2, 2, 2)), atol=1e-14)


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


def test_complex_det_and_slogdet():
    a = np.array([[2 + 1j, 1 - 2j], [1j, 3 - 1j]])
    expected = a[0, 0] * a[1, 1] - a[0, 1] * a[1, 0]
    assert_allclose(np.linalg.det(a), expected, atol=1e-12)
    sign, logabs = np.linalg.slogdet(a)
    assert_allclose(sign * np.exp(logabs), expected, atol=1e-12)
    assert_allclose(np.abs(sign), 1.0, atol=1e-12)
    singular = np.array([[1 + 1j, 2 + 2j], [2 + 2j, 4 + 4j]])
    sign, logabs = np.linalg.slogdet(singular)
    assert sign == 0
    assert logabs == -np.inf


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


def test_complex_pinv_and_lstsq():
    a = np.array([[1 + 1j, 2], [0, 1 - 1j], [2j, 3]])
    pseudo = np.linalg.pinv(a)
    assert_allclose(a @ pseudo @ a, a, atol=1e-10)
    b = np.array([1j, 2, 3 + 1j])
    x, residuals, rank, values = np.linalg.lstsq(a, b)
    assert rank == 2
    assert values.shape == (2,)
    assert_allclose(x, pseudo @ b, atol=1e-10)
    assert_allclose(residuals, [np.sum(np.abs(b - a @ x) ** 2)], atol=1e-10)


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


def test_complex_qr_reconstructs_and_is_unitary():
    a = np.array([[1 + 1j, 2], [0, 1 - 1j], [2j, 3]])
    q, r = np.linalg.qr(a)
    assert q.shape == (3, 2)
    assert_allclose(q.conj().T @ q, np.eye(2), atol=1e-12)
    assert_allclose(q @ r, a, atol=1e-12)
    q_full, r_full = np.linalg.qr(a, mode="complete")
    assert q_full.shape == (3, 3)
    assert_allclose(q_full.conj().T @ q_full, np.eye(3), atol=1e-12)
    assert_allclose(q_full @ r_full, a, atol=1e-12)
    assert_allclose(np.linalg.qr(a, mode="r"), r, atol=1e-12)


def test_cholesky_is_lower_triangular_factor():
    a = spd_matrix()
    lower = np.linalg.cholesky(a)
    assert_array_equal(np.triu(lower, 1), np.zeros((3, 3)))
    assert np.all(np.diag(lower) > 0)
    assert_allclose(lower @ lower.T, a, atol=1e-12)
    assert_allclose(lower[0, 0], 2.0, rtol=1e-15)


def test_complex_cholesky_reconstructs_hermitian_matrix():
    a = np.array([[5.0, 1 - 2j], [1 + 2j, 4.0]])
    lower = np.linalg.cholesky(a)
    assert_allclose(lower @ lower.conj().T, a, atol=1e-12)
    assert_array_equal(np.triu(lower, 1), np.zeros((2, 2)))
    with pytest.raises(np.linalg.LinAlgError):
        np.linalg.cholesky(np.array([[1.0, 2j], [-2j, 1.0]]))


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


def test_complex_hermitian_eigh_and_eigvalsh():
    a = np.array([[2.0, 1 - 2j], [1 + 2j, 4.0]])
    w, v = np.linalg.eigh(a)
    assert w.dtype == np.float64
    assert v.dtype == np.complex128
    assert_allclose(a @ v, v * w, atol=1e-12)
    assert_allclose(v.conj().T @ v, np.eye(2), atol=1e-12)
    assert_allclose(w, np.linalg.eigvalsh(a), atol=1e-12)
    upper = np.array([[2.0, 1 - 2j], [999.0, 4.0]], dtype=complex)
    assert_allclose(np.linalg.eigvalsh(upper, UPLO="U"), w, atol=1e-12)


def test_complex_hermitian_repeated_eigenvalues():
    a = np.eye(3, dtype=complex) * (2 + 0j)
    w, v = np.linalg.eigh(a)
    assert_array_equal(w, [2.0, 2.0, 2.0])
    assert_allclose(v.conj().T @ v, np.eye(3), atol=1e-12)


def sorted_eigenvalues(w):
    # Eigenvalue order is not part of the contract, so comparisons sort by (real, imag) first.
    order = np.lexsort((w.imag, w.real))
    return w[order]


def test_eig_general_matrix_satisfies_eigenvector_equation():
    # Eigenvectors are only defined up to sign/phase, so this checks the defining relation
    # A @ v == w * v column-by-column rather than comparing v to a fixed reference.
    a = general_matrix()
    w, v = np.linalg.eig(a)
    assert w.dtype == np.complex128
    assert v.dtype == np.complex128
    assert_allclose(a @ v, v * w, atol=1e-10)
    assert_allclose(sorted_eigenvalues(w), sorted_eigenvalues(np.linalg.eigvals(a)), atol=1e-10)
    assert_allclose(np.sum(w).real, np.trace(a), atol=1e-10)
    assert_allclose(np.prod(w).real, np.linalg.det(a), atol=1e-10)


def test_eig_rotation_matrix_has_complex_conjugate_pair():
    a = np.array([[0.0, -1.0], [1.0, 0.0]])
    w, v = np.linalg.eig(a)
    assert_allclose(sorted_eigenvalues(w), sorted_eigenvalues(np.array([1j, -1j])), atol=1e-12)
    assert_allclose(a @ v, v * w, atol=1e-12)


def test_complex_eig_and_eigvals_satisfy_eigenvector_equation():
    a = np.array([[1 + 2j, 1 - 1j], [0, 3 - 1j]])
    w, v = np.linalg.eig(a)
    assert_allclose(a @ v, v * w, atol=1e-10)
    assert_allclose(sorted_eigenvalues(w), sorted_eigenvalues(np.array([1 + 2j, 3 - 1j])), atol=1e-10)
    assert_allclose(sorted_eigenvalues(np.linalg.eigvals(a)), sorted_eigenvalues(w), atol=1e-10)


def test_complex_eig_coupled_matrix():
    a = np.array([[1 + 1j, 2 - 1j], [3j, 4 - 2j]])
    w, v = np.linalg.eig(a)
    assert_allclose(a @ v, v * w, atol=1e-9)
    assert_allclose(np.sum(w), np.trace(a), atol=1e-9)
    assert_allclose(np.prod(w), np.linalg.det(a), atol=1e-9)
    larger = np.array([[1 + 1j, 2, 0], [0, 3 - 2j, 1j], [1, 0, -1 + 0.5j]])
    w, v = np.linalg.eig(larger)
    assert_allclose(larger @ v, v * w, atol=1e-8)
    assert_allclose(np.sum(w), np.trace(larger), atol=1e-8)
    rotation = np.array([[0.0, -1.0], [1.0, 0.0]], dtype=complex)
    w, v = np.linalg.eig(rotation)
    assert_allclose(np.abs(w), [1, 1], atol=1e-9)
    assert_allclose(np.sum(w), 0, atol=1e-9)
    assert_allclose(np.prod(w), 1, atol=1e-9)
    assert_allclose(rotation @ v, v * w, atol=1e-9)


def test_complex_eig_repeated_real_eigenvalue():
    a = np.array([[2 + 0j, 1], [0, 2]])
    w, v = np.linalg.eig(a)
    assert_allclose(w, [2, 2], atol=1e-8)
    assert_allclose(a @ v, v * w, atol=1e-8)
    repeated = np.diag(np.array([1 + 0j, 1, 2]))
    w, v = np.linalg.eig(repeated)
    assert_allclose(sorted_eigenvalues(w), [1, 1, 2], atol=1e-8)
    assert_allclose(repeated @ v, v * w, atol=1e-8)


def test_eig_defective_matrix_repeats_eigenvalue():
    # A single non-trivial Jordan block: eigenvalue 2 with algebraic multiplicity 2 but only one
    # independent eigenvector, so only the eigenvector equation is checked, not orthogonality.
    a = np.array([[2.0, 1.0], [0.0, 2.0]])
    w, v = np.linalg.eig(a)
    assert_allclose(sorted_eigenvalues(w), [2.0, 2.0], atol=1e-8)
    for k in range(2):
        assert_allclose(a @ v[:, k], w[k] * v[:, k], atol=1e-8)


def test_eig_non_square_raises_linalg_error():
    with pytest.raises(np.linalg.LinAlgError):
        np.linalg.eig(np.ones((2, 3)))


def test_eigvals_matches_eig_without_vectors():
    a = spd_matrix()
    w, _ = np.linalg.eig(a)
    assert_allclose(sorted_eigenvalues(w), sorted_eigenvalues(np.linalg.eigvals(a)), atol=1e-10)


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


def test_complex_svd_reconstructs_and_preserves_dtype():
    a = np.array([[1 + 1j, 2], [0, 1 - 1j], [2j, 3]], dtype=np.complex64)
    u, s, vh = np.linalg.svd(a, full_matrices=False)
    assert u.shape == (3, 2)
    assert s.dtype == np.float32
    assert vh.dtype == np.complex64
    assert_allclose(u @ np.diag(s) @ vh, a, atol=1e-5)
    assert_allclose(u.conj().T @ u, np.eye(2), atol=1e-5)
    assert_allclose(vh @ vh.conj().T, np.eye(2), atol=1e-5)
    assert_allclose(np.linalg.svd(a, compute_uv=False), s, atol=1e-5)
    assert_allclose(np.linalg.svdvals(a), s, atol=1e-5)


def test_complex_svd_rank_deficient_matrix():
    a = np.array([[1 + 1j, 2 + 2j], [2 - 1j, 4 - 2j]])
    u, s, vh = np.linalg.svd(a)
    assert s[1] < 1e-7
    assert_allclose(u @ np.diag(s) @ vh, a, atol=1e-10)
    assert_allclose(u.conj().T @ u, np.eye(2), atol=1e-10)
    assert_allclose(a @ np.linalg.pinv(a) @ a, a, atol=1e-10)
    assert np.linalg.matrix_rank(a) == 1


def test_decompositions_return_float64_for_int_input():
    a = np.array([[2, 1], [1, 2]])
    assert np.linalg.inv(a).dtype == np.float64
    assert np.linalg.det(a).dtype == np.float64
    assert np.linalg.eigh(a)[0].dtype == np.float64
    assert np.linalg.svd(a)[1].dtype == np.float64


def test_vecdot_sums_over_an_axis_and_conjugates_the_first_operand():
    a = np.array([[1.0, 2.0], [3.0, 4.0]])
    assert np.vecdot(a, np.array([1.0, 1.0])).tolist() == [3.0, 7.0]
    assert np.vecdot(a, a, axis=0).tolist() == [10.0, 20.0]
    assert np.vecdot(np.array([1 + 1j, 2]), np.array([1j, 1])) == 3 + 1j
    assert np.vecdot([1, 2], [3, 4]) == 11


def test_vecdot_rejects_mismatched_core_dimensions():
    with pytest.raises(ValueError):
        np.vecdot(np.ones((2, 3)), np.ones(3), axis=0)


def test_einsum_matmul_matches_dot_and_matmul():
    a = np.arange(6).reshape(2, 3)
    b = np.arange(12).reshape(3, 4)
    assert_allclose(np.einsum("ij,jk->ik", a, b), np.dot(a, b))
    assert np.einsum("ij,jk->ik", a, b).dtype == np.dot(a, b).dtype
    batch_a = np.arange(24).reshape(2, 3, 4)
    batch_b = np.arange(24).reshape(2, 4, 3)
    assert_allclose(np.einsum("bij,bjk->bik", batch_a, batch_b), batch_a @ batch_b)


def test_einsum_trace_and_diagonal_match_the_named_functions():
    m = np.arange(9).reshape(3, 3)
    assert np.einsum("ii", m) == np.trace(m)
    assert_array_equal(np.einsum("ii->i", m), np.diagonal(m))


def test_einsum_outer_product_matches_outer():
    a = np.array([1, 2, 3])
    b = np.array([4, 5])
    assert_array_equal(np.einsum("i,j->ij", a, b), np.outer(a, b))


def test_einsum_transpose_reorders_axes_without_a_copyable_operation():
    a = np.arange(6).reshape(2, 3)
    assert_array_equal(np.einsum("ij->ji", a), a.T)


def test_einsum_implicit_mode_sums_unlisted_repeated_indices():
    a = np.arange(6).reshape(2, 3)
    b = np.arange(12).reshape(3, 4)
    # No '->': every index appearing exactly once survives, sorted; 'j' is contracted away.
    assert_array_equal(np.einsum("ij,jk", a, b), np.dot(a, b))
    assert_array_equal(np.einsum("ii", np.arange(9).reshape(3, 3)), np.trace(np.arange(9).reshape(3, 3)))


def test_einsum_three_operands_contract_left_to_right():
    a = np.arange(6).reshape(2, 3)
    b = np.arange(12).reshape(3, 4)
    c = np.arange(8).reshape(4, 2)
    assert_allclose(np.einsum("ij,jk,ki->", a, b, c), np.trace(a @ b @ c))


def test_einsum_integer_operands_keep_integer_dtype():
    a = np.arange(6).reshape(2, 3)
    b = np.arange(12).reshape(3, 4)
    result = np.einsum("ij,jk->ik", a, b)
    assert result.dtype == np.int64
    assert_array_equal(result, a @ b)


def test_einsum_optimize_flag_is_accepted_and_ignored():
    a = np.arange(6).reshape(2, 3)
    b = np.arange(12).reshape(3, 4)
    assert_array_equal(np.einsum("ij,jk->ik", a, b, optimize=True), a @ b)
    assert_array_equal(np.einsum("ij,jk->ik", a, b, optimize="optimal"), a @ b)


def test_einsum_rejects_repeated_output_indices():
    with pytest.raises(ValueError):
        np.einsum("ii->ii", np.eye(3))


def test_einsum_rejects_mismatched_operand_shapes():
    with pytest.raises(ValueError):
        np.einsum("ij,jk->ik", np.zeros((2, 3)), np.zeros((5, 4)))


def test_einsum_rejects_wrong_operand_count_for_the_subscripts():
    with pytest.raises(ValueError):
        np.einsum("ij,jk->ik", np.zeros((2, 3)))
