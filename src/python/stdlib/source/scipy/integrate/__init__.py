"""shellsim's ``scipy.integrate``: quadrature of samples and of Python callables.

Implemented: ``trapezoid``, ``cumulative_trapezoid`` and ``simpson`` over sampled data, and the
adaptive ``quad`` with ``dblquad``, ``tplquad`` and ``nquad`` built on it. The other SciPy names
raise ``NotImplementedError`` when accessed.
"""

from scipy.integrate._quadpack import IntegrationWarning, dblquad, nquad, quad, tplquad
from scipy.integrate._quadrature import cumulative_trapezoid, simpson, trapezoid

__all__ = [
    "IntegrationWarning",
    "cumulative_trapezoid",
    "dblquad",
    "nquad",
    "quad",
    "simpson",
    "tplquad",
    "trapezoid",
]

_UNSUPPORTED = {
    "BDF",
    "DOP853",
    "DenseOutput",
    "LSODA",
    "ODEintWarning",
    "OdeSolution",
    "OdeSolver",
    "RK23",
    "RK45",
    "Radau",
    "complex_ode",
    "cubature",
    "cumulative_simpson",
    "fixed_quad",
    "lebedev_rule",
    "newton_cotes",
    "nsum",
    "ode",
    "odeint",
    "qmc_quad",
    "quad_vec",
    "romb",
    "solve_bvp",
    "solve_ivp",
    "tanhsinh",
}


def __getattr__(name):
    if name in _UNSUPPORTED:
        raise NotImplementedError(f"scipy.integrate.{name} is not supported by shellsim's SciPy")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
