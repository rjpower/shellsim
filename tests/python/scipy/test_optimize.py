# Portable SciPy semantics. Expectations checked against SciPy 1.18.1 and NumPy 2.5.3 on
# CPython 3.14.4.
# Scope: scipy.optimize's minimize (Nelder-Mead, BFGS), minimize_scalar, the scalar root finders
# (root_scalar, brentq, bisect, newton) and curve_fit.
# Every solver here is an independent reimplementation of a published algorithm (Nelder-Mead
# 1965, BFGS 1970, Brent 1973, Levenberg-Marquardt), not a port of SciPy's MINPACK-based code,
# so tests check the solution against its known analytic value with a tolerance loose enough to
# also hold for SciPy's own iteration path, rather than the iteration count or internal state.

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy import optimize as so


def rosenbrock(x):
    x = np.asarray(x)
    return np.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2)


# -------------------------------------------------------------------------------------------
# minimize
# -------------------------------------------------------------------------------------------


def test_minimize_rosenbrock_with_nelder_mead():
    result = so.minimize(rosenbrock, [-1.2, 1.0], method="Nelder-Mead")
    assert result.success
    assert_allclose(result.x, [1.0, 1.0], rtol=1e-3, atol=1e-3)
    assert result.fun < 1e-6


def test_minimize_rosenbrock_with_bfgs():
    result = so.minimize(rosenbrock, [-1.2, 1.0], method="BFGS")
    assert result.success
    assert_allclose(result.x, [1.0, 1.0], rtol=1e-4, atol=1e-4)
    assert result.fun < 1e-8


def test_minimize_defaults_to_bfgs():
    result = so.minimize(rosenbrock, [-1.2, 1.0])
    assert_allclose(result.x, [1.0, 1.0], rtol=1e-4, atol=1e-4)


def test_minimize_quadratic_with_analytic_jacobian():
    def fun(x):
        return (x[0] - 3.0) ** 2 + (x[1] + 1.0) ** 2

    def jac(x):
        return np.array([2.0 * (x[0] - 3.0), 2.0 * (x[1] + 1.0)])

    result = so.minimize(fun, [0.0, 0.0], jac=jac, method="BFGS")
    assert result.success
    assert_allclose(result.x, [3.0, -1.0], atol=1e-6)
    assert_allclose(result.jac, [0.0, 0.0], atol=1e-5)
    assert result.nit > 0 and result.nfev > 0


# -------------------------------------------------------------------------------------------
# minimize_scalar
# -------------------------------------------------------------------------------------------


def test_minimize_scalar_on_a_parabola():
    result = so.minimize_scalar(lambda x: (x - 2.0) ** 2)
    assert_allclose(result.x, 2.0, atol=1e-6)
    assert result.fun < 1e-10


def test_minimize_scalar_bounded():
    result = so.minimize_scalar(lambda x: (x - 2.0) ** 2, bounds=(0.0, 1.0))
    assert_allclose(result.x, 1.0, atol=1e-4)
    result = so.minimize_scalar(lambda x: (x - 2.0) ** 2, bounds=(0.0, 5.0))
    assert_allclose(result.x, 2.0, atol=1e-4)


# -------------------------------------------------------------------------------------------
# Root finding
# -------------------------------------------------------------------------------------------

CUBE_ROOT_OF_2 = 2.0 ** (1.0 / 3.0)


def test_root_finders_solve_x_cubed_minus_2():
    assert_allclose(so.brentq(lambda x: x**3 - 2, 1, 2), CUBE_ROOT_OF_2, atol=1e-9)
    assert_allclose(so.bisect(lambda x: x**3 - 2, 1, 2), CUBE_ROOT_OF_2, atol=1e-6)
    assert_allclose(so.newton(lambda x: x**3 - 2, 1.0), CUBE_ROOT_OF_2, atol=1e-6)
    assert_allclose(
        so.newton(lambda x: x**3 - 2, 1.0, fprime=lambda x: 3 * x**2), CUBE_ROOT_OF_2, atol=1e-9
    )


def test_root_scalar_dispatches_by_argument():
    result = so.root_scalar(lambda x: x**3 - 2, bracket=(1, 2))
    assert result.converged
    assert_allclose(result.root, CUBE_ROOT_OF_2, atol=1e-9)
    assert result.iterations > 0 and result.function_calls > 0
    bisected = so.root_scalar(lambda x: x**3 - 2, bracket=(1, 2), method="bisect")
    assert_allclose(bisected.root, CUBE_ROOT_OF_2, atol=1e-6)
    newton_result = so.root_scalar(
        lambda x: x**3 - 2, x0=1.0, fprime=lambda x: 3 * x**2
    )
    assert newton_result.converged
    assert_allclose(newton_result.root, CUBE_ROOT_OF_2, atol=1e-9)
    secant_result = so.root_scalar(lambda x: x**3 - 2, x0=1.0)
    assert_allclose(secant_result.root, CUBE_ROOT_OF_2, atol=1e-6)


def test_bracket_without_sign_change_raises_value_error():
    with pytest.raises(ValueError):
        so.brentq(lambda x: x**3 - 2, 2, 3)
    with pytest.raises(ValueError):
        so.bisect(lambda x: x**3 - 2, 2, 3)
    with pytest.raises(ValueError):
        so.root_scalar(lambda x: x**3 - 2, bracket=(2, 3))


# -------------------------------------------------------------------------------------------
# curve_fit
# -------------------------------------------------------------------------------------------


def test_curve_fit_recovers_noiseless_exponential_parameters():
    def model(x, a, b):
        return a * np.exp(-b * x)

    xdata = np.linspace(0.0, 4.0, 50)
    ydata = model(xdata, 2.5, 1.3)
    popt, pcov = so.curve_fit(model, xdata, ydata, p0=[1.0, 1.0])
    assert_allclose(popt, [2.5, 1.3], rtol=1e-6)
    assert pcov.shape == (2, 2)
    assert np.all(np.isfinite(pcov))
    # p0 can be omitted; the parameter count is inferred from `model`'s signature.
    popt_auto, _ = so.curve_fit(model, xdata, ydata)
    assert_allclose(popt_auto, [2.5, 1.3], rtol=1e-6)
