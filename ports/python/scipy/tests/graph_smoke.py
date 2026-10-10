"""Run numerical and callback behavior in the installed upstream SciPy guest."""

import numpy as np
from scipy import integrate, linalg, sparse
from scipy.sparse import linalg as sparse_linalg

matrix = np.array([[3.0, 1.0], [1.0, 2.0]])
rhs = np.array([9.0, 8.0])
np.testing.assert_allclose(linalg.solve(matrix, rhs), [2.0, 3.0], rtol=1e-12)
np.testing.assert_allclose(linalg.blas.dgemm(1.0, matrix, matrix), matrix @ matrix)
complex_matrix = matrix.astype(complex) + 1j * np.eye(2)
np.testing.assert_allclose(complex_matrix @ linalg.solve(complex_matrix, rhs), rhs, atol=1e-12)
try:
    linalg.solve(matrix, np.ones(3))
except ValueError:
    pass
else:
    raise AssertionError("incompatible solve dimensions accepted")
value, error = integrate.quad(lambda x: x * x, 0.0, 1.0)
assert abs(value - 1.0 / 3.0) < 1e-12
assert error < 1e-10
csr = sparse.csr_matrix(matrix)
np.testing.assert_allclose(csr @ rhs, matrix @ rhs)
np.testing.assert_allclose(sparse_linalg.spsolve(csr, rhs), [2.0, 3.0], rtol=1e-12)
print("SciPy BLAS, real/complex solve, integration callback and sparse solve passed")
