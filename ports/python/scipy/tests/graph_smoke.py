"""Run numerical and callback behavior in the installed upstream SciPy guest."""

import numpy as np
from scipy import fft, integrate, interpolate, linalg, optimize, sparse
from scipy.sparse import linalg as sparse_linalg

samples = np.arange(128, dtype=float).reshape(4, 32)
samples = np.sin(samples / 7.0) + 0.25j * np.cos(samples / 11.0)
spectrum = fft.fft(samples, axis=-1, workers=2)
np.testing.assert_allclose(spectrum, np.fft.fft(samples, axis=-1), rtol=1e-12, atol=1e-12)
np.testing.assert_allclose(fft.ifft(spectrum, axis=-1, workers=2), samples, atol=1e-12)

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
triangular, vectors = linalg.schur(matrix)
np.testing.assert_allclose(vectors @ triangular @ vectors.T, matrix, atol=1e-12)
other = np.diag([4.0, 5.0])
solution = linalg.solve_sylvester(matrix, other, np.eye(2))
np.testing.assert_allclose(matrix @ solution + solution @ other, np.eye(2), atol=1e-12)
linear = optimize.linprog([-1.0, -2.0], A_ub=[[1.0, 1.0]], b_ub=[3.0])
assert linear.success
np.testing.assert_allclose(linear.fun, -6.0, atol=1e-10)
assert np.all(linear.x >= -1e-10)
assert linear.x.sum() <= 3.0 + 1e-10
minimum = optimize.minimize(
    lambda x: float(x @ x),
    np.array([2.0, -3.0]),
    jac=lambda x: 2 * x,
    hess=lambda x: 2 * np.eye(len(x)),
    method="trust-krylov",
)
assert minimum.success
np.testing.assert_allclose(minimum.x, 0.0, atol=1e-8)
eigen_matrix = sparse.diags(np.arange(1.0, 7.0), format="csr")
eigenvalues, eigenvectors = sparse_linalg.eigs(eigen_matrix, k=1, which="LM", v0=np.ones(6))
np.testing.assert_allclose(eigenvalues, [6.0], atol=1e-10)
np.testing.assert_allclose(eigen_matrix @ eigenvectors, eigenvectors * eigenvalues, atol=1e-10)
points = np.linspace(0.0, 1.0, 8)
spline = interpolate.splrep(points, points**3, s=0)
np.testing.assert_allclose(interpolate.splev(points, spline), points**3, atol=1e-12)
_, _, info = linalg.lapack.dgetrf(np.zeros((2, 2)))
assert info > 0
try:
    linalg.lapack.dgesv(matrix, np.ones((3, 1)))
except ValueError:
    pass
else:
    raise AssertionError("invalid LAPACK right-hand-side dimensions accepted")
print("SciPy solve, Schur, Sylvester, optimization, sparse eigenvalues and splines passed")
