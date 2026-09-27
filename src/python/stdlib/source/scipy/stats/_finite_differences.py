"""Central finite differences, following SciPy 1.18's ``scipy/stats/_finite_differences.py``.

``rv_continuous`` uses ``_derivative`` for the density of a distribution that defines only its
CDF. Weights for orders SciPy does not tabulate come from ``numpy.linalg.inv`` where SciPy uses
``scipy.linalg.inv``.
"""

from numpy import arange, array, hstack, newaxis, prod
from numpy.linalg import inv


def _central_diff_weights(Np, ndiv=1):
    """Weights for an ``Np``-point central derivative of order ``ndiv``."""
    if Np < ndiv + 1:
        raise ValueError("Number of points must be at least the derivative order + 1.")
    if Np % 2 == 0:
        raise ValueError("The number of points must be odd.")

    ho = Np >> 1
    x = arange(-ho, ho + 1.0)
    x = x[:, newaxis]
    X = x**0.0
    for k in range(1, Np):
        X = hstack([X, x**k])
    w = prod(arange(1, ndiv + 1), axis=0) * inv(X)[ndiv]
    return w


def _derivative(func, x0, dx=1.0, n=1, args=(), order=3):
    """The ``n``-th derivative of ``func`` at ``x0`` from ``order`` points spaced ``dx`` apart."""
    if order < n + 1:
        raise ValueError(
            "'order' (the number of points used to compute the derivative), "
            "must be at least the derivative order 'n' + 1."
        )
    if order % 2 == 0:
        raise ValueError(
            "'order' (the number of points used to compute the derivative) must be odd."
        )
    # pre-computed for n=1 and 2 and low-order for speed.
    if n == 1:
        if order == 3:
            weights = array([-1, 0, 1]) / 2.0
        elif order == 5:
            weights = array([1, -8, 0, 8, -1]) / 12.0
        elif order == 7:
            weights = array([-1, 9, -45, 0, 45, -9, 1]) / 60.0
        elif order == 9:
            weights = array([3, -32, 168, -672, 0, 672, -168, 32, -3]) / 840.0
        else:
            weights = _central_diff_weights(order, 1)
    elif n == 2:
        if order == 3:
            weights = array([1, -2.0, 1])
        elif order == 5:
            weights = array([-1, 16, -30, 16, -1]) / 12.0
        elif order == 7:
            weights = array([2, -27, 270, -490, 270, -27, 2]) / 180.0
        elif order == 9:
            weights = array([-9, 128, -1008, 8064, -14350, 8064, -1008, 128, -9]) / 5040.0
        else:
            weights = _central_diff_weights(order, 2)
    else:
        weights = _central_diff_weights(order, n)
    val = 0.0
    ho = order >> 1
    for k in range(order):
        val += weights[k] * func(x0 + (k - ho) * dx, *args)
    return val / prod((dx,) * n, axis=0)
