"""shellsim's ``scipy.integrate``: quadrature of samples and of Python callables.

Implemented: ``trapezoid``, ``cumulative_trapezoid`` and ``simpson`` over sampled data, and the
adaptive ``quad`` with ``dblquad``, ``tplquad`` and ``nquad`` built on it.
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
