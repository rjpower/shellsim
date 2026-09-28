# Portable SciPy semantics. Expectations checked against SciPy 1.18.1 and NumPy 2.5.3 on
# CPython 3.14.4.
# Scope: scipy.integrate's quad (and the dblquad wrapper), trapezoid/cumulative_trapezoid/
# simpson, and solve_ivp/odeint.
# quad's Gauss-Kronrod adaptive bisection with Wynn extrapolation is an independent
# reimplementation, not a port of QUADPACK, so results are checked against closed forms with
# tolerances rather than against SciPy's literal bits; SciPy meets the same closed forms to the
# same tolerances. solve_ivp/odeint likewise integrate an independent Dormand-Prince RK45
# stepper, checked against closed-form solutions at tight tolerances so that the cubic-Hermite
# dense output (rather than SciPy's own quartic interpolant) does not dominate the comparison.

import warnings

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy import integrate as si


def recorded(function, *args, **kwargs):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = function(*args, **kwargs)
    return result, [warning.category for warning in caught]


# -------------------------------------------------------------------------------------------
# quad, dblquad
# -------------------------------------------------------------------------------------------


def test_quad_polynomial_matches_closed_form():
    value, error = si.quad(lambda x: x**2, 0, 1)
    assert_allclose(value, 1 / 3, atol=1e-12)
    assert error < 1e-8
    value, _ = si.quad(lambda x: 3 * x**2 - 2 * x + 1, -1, 2, args=())
    assert_allclose(value, 9.0, atol=1e-10)
    # `args` passes extra parameters through to the integrand.
    value, _ = si.quad(lambda x, a, b: a * x + b, 0, 1, args=(2.0, 1.0))
    assert_allclose(value, 2.0, atol=1e-10)


def test_quad_oscillatory_matches_closed_form():
    # integral of cos(k*x) from 0 to b is sin(k*b)/k.
    k, b = 25.0, 3.0
    value, error = si.quad(lambda x: np.cos(k * x), 0, b)
    assert_allclose(value, np.sin(k * b) / k, atol=1e-8)
    assert error < 1e-6


def test_quad_endpoint_singularity_uses_extrapolation():
    # integral of 1/sqrt(x) from 0 to 1 is 2; the Gauss-Kronrod error estimate alone stalls at
    # this integrable singularity, so reaching 1e-10 needs Wynn extrapolation, with no warning.
    (value, error), caught = recorded(si.quad, lambda x: 1 / np.sqrt(x), 0, 1)
    assert_allclose(value, 2.0, atol=1e-9)
    assert caught == []


def test_quad_infinite_limits_match_closed_form():
    value, _ = si.quad(np.exp, -np.inf, 0)
    assert_allclose(value, 1.0, atol=1e-7)
    value, _ = si.quad(lambda x: np.exp(-(x**2)), -np.inf, np.inf)
    assert_allclose(value, np.sqrt(np.pi), atol=1e-7)
    value, _ = si.quad(lambda x: np.exp(-x), 1, np.inf)
    assert_allclose(value, np.exp(-1), atol=1e-7)


def test_quad_warns_when_tolerance_is_not_met():
    # A divergent integral cannot meet any tolerance; QUADPACK's transform still returns a
    # finite (meaningless) number, flagged with a warning.
    _, caught = recorded(si.quad, lambda x: x, 1, np.inf)
    assert caught == [si.IntegrationWarning]
    assert issubclass(si.IntegrationWarning, UserWarning)


def test_quad_rejects_bad_input():
    with pytest.raises(ValueError):
        si.quad(1.0, 0, 1)
    with pytest.raises(ValueError):
        si.quad(lambda x: x, 0, 1, limit=0)
    with pytest.raises(TypeError):
        si.quad(lambda x: "not a number", 0, 1)


def test_dblquad_matches_closed_form():
    value, _ = si.dblquad(lambda y, x: x * y, 0, 1, 0, lambda x: x)
    assert_allclose(value, 0.125, atol=1e-10)
    value, _ = si.dblquad(lambda y, x: 1.0, 0, 2, 0, 3)
    assert_allclose(value, 6.0, atol=1e-10)


# -------------------------------------------------------------------------------------------
# trapezoid, cumulative_trapezoid, simpson
# -------------------------------------------------------------------------------------------


def test_trapezoid_and_cumulative_trapezoid():
    assert si.trapezoid([1.0, 2.0, 3.0]) == 4.0
    assert si.trapezoid([1.0, 2.0, 4.0], dx=2.0) == 9.0
    assert si.trapezoid([1.0, 2.0, 4.0], x=[0.0, 1.0, 3.0]) == 7.5
    assert_allclose(si.cumulative_trapezoid([1.0, 2.0, 3.0]), [1.5, 4.0])
    assert_allclose(si.cumulative_trapezoid([1.0, 2.0, 3.0], initial=0), [0.0, 1.5, 4.0])
    y = np.array([[1.0, 2.0, 3.0], [2.0, 4.0, 6.0]])
    assert_allclose(si.cumulative_trapezoid(y, axis=1, initial=0), [[0.0, 1.5, 4.0], [0.0, 3.0, 8.0]])
    assert_allclose(
        si.cumulative_trapezoid(y, x=[0.0, 1.0, 3.0], axis=1), [[1.5, 6.5], [3.0, 13.0]]
    )
    with pytest.raises(ValueError):
        si.cumulative_trapezoid([1.0, 2.0, 3.0], initial=1.0)


def test_simpson_odd_and_even_sample_counts():
    # Odd sample count: exact for a cubic, which composite Simpson integrates exactly.
    x = np.linspace(0.0, 2.0, 5)
    assert_allclose(si.simpson(x**3, x=x), 4.0, atol=1e-12)
    # An even sample count leaves one interval short of a full Simpson pair; Cartwright's
    # correction covers it with the parabola through the last three points, which is not
    # exact for a cubic but converges to the closed form as the spacing shrinks.
    x_even = np.linspace(0.0, 2.0, 50)
    assert_allclose(si.simpson(x_even**3, x=x_even), 4.0, atol=1e-5)
    assert si.simpson([1.0, 2.0]) == si.trapezoid([1.0, 2.0])
    assert si.simpson([5.0]) == 0.0
    with pytest.raises(IndexError):
        si.simpson(np.array([]))


def test_simpson_axis_and_uniform_spacing():
    y = np.stack([np.linspace(0.0, 2.0, 5) ** 2, np.linspace(0.0, 4.0, 5) ** 2])
    along_rows = si.simpson(y, dx=0.5, axis=1)
    assert_allclose(along_rows[0], si.simpson(y[0], dx=0.5))
    assert_allclose(along_rows[1], si.simpson(y[1], dx=0.5))
    assert_allclose(si.simpson(y, dx=0.5, axis=-1), along_rows)


# -------------------------------------------------------------------------------------------
# solve_ivp, odeint
# -------------------------------------------------------------------------------------------


def test_solve_ivp_exponential_decay():
    t_eval = np.linspace(0.0, 5.0, 6)
    result = si.solve_ivp(lambda t, y: -y, (0.0, 5.0), [1.0], t_eval=t_eval, rtol=1e-9, atol=1e-11)
    assert result.success and result.status == 0
    assert result.y.shape == (1, len(t_eval))
    assert_allclose(result.y[0], np.exp(-t_eval), rtol=1e-6, atol=1e-8)
    assert isinstance(result.nfev, int) and result.nfev > 0


def test_solve_ivp_harmonic_oscillator():
    def f(t, y):
        return [y[1], -y[0]]

    t_eval = np.linspace(0.0, 10.0, 11)
    result = si.solve_ivp(f, (0.0, 10.0), [1.0, 0.0], t_eval=t_eval, rtol=1e-9, atol=1e-11)
    assert result.y.shape == (2, len(t_eval))
    assert_allclose(result.y[0], np.cos(t_eval), rtol=1e-6, atol=1e-7)
    assert_allclose(result.y[1], -np.sin(t_eval), rtol=1e-6, atol=1e-7)


def test_solve_ivp_without_t_eval_reaches_the_endpoint():
    result = si.solve_ivp(lambda t, y: -y, (0.0, 2.0), [1.0])
    assert result.t[0] == 0.0 and result.t[-1] == 2.0
    assert result.y.shape == (1, len(result.t))
    assert_allclose(result.y[0, -1], np.exp(-2.0), rtol=1e-2)


def test_solve_ivp_passes_args_and_supports_backward_integration():
    result = si.solve_ivp(
        lambda t, y, k: -k * y, (0.0, 1.0), [1.0], args=(2.0,), t_eval=[0.0, 0.5, 1.0],
        rtol=1e-9, atol=1e-11,
    )
    assert_allclose(result.y[0], np.exp(-2.0 * np.array([0.0, 0.5, 1.0])), rtol=1e-6, atol=1e-8)
    backward = si.solve_ivp(
        lambda t, y: -y, (1.0, 0.0), [np.exp(-1.0)], t_eval=[1.0, 0.5, 0.0], rtol=1e-9, atol=1e-11
    )
    assert_allclose(backward.y[0], np.exp(-np.array([1.0, 0.5, 0.0])), rtol=1e-6, atol=1e-8)


def test_odeint_matches_solve_ivp_convention():
    t = np.linspace(0.0, 5.0, 6)
    y = si.odeint(lambda y, t: -y, 1.0, t)
    assert y.shape == (len(t), 1)
    assert_allclose(y[:, 0], np.exp(-t), rtol=1e-6, atol=1e-8)
