"""shellsim's ``scipy.interpolate``: one-dimensional interpolation.

Implemented: ``interp1d`` (every kind, including spline orders above 3), the piecewise
polynomial ``PPoly``, and the cubic families built on it: ``CubicHermiteSpline``,
``CubicSpline`` (not-a-knot, natural, clamped, periodic and explicit derivative end
conditions) and ``PchipInterpolator``/``pchip_interpolate``. Everything is composed from NumPy
array operations; the only linear solves are banded ones through ``scipy.linalg.solve_banded``,
so building a spline costs time linear in the number of points.

Results agree with SciPy to rounding. ``interp1d``'s linear kind reproduces SciPy bit for bit;
the spline constructions solve the same systems but may round differently in the last bits.
B-spline objects, multivariate and scattered-data interpolation, smoothing splines and the
FITPACK wrappers are not modeled and raise ``NotImplementedError`` when accessed.
"""

import math
import operator

import numpy as np

from scipy.linalg import solve_banded

__all__ = [
    "CubicHermiteSpline",
    "CubicSpline",
    "PPoly",
    "PchipInterpolator",
    "interp1d",
    "pchip",
    "pchip_interpolate",
]

_UNSUPPORTED = {
    "AAA",
    "Akima1DInterpolator",
    "BPoly",
    "BSpline",
    "BarycentricInterpolator",
    "BivariateSpline",
    "CloughTocher2DInterpolator",
    "FloaterHormannInterpolator",
    "InterpolatedUnivariateSpline",
    "KroghInterpolator",
    "LSQBivariateSpline",
    "LSQSphereBivariateSpline",
    "LSQUnivariateSpline",
    "LinearNDInterpolator",
    "NdBSpline",
    "NdPPoly",
    "NearestNDInterpolator",
    "RBFInterpolator",
    "Rbf",
    "RectBivariateSpline",
    "RectSphereBivariateSpline",
    "RegularGridInterpolator",
    "SmoothBivariateSpline",
    "SmoothSphereBivariateSpline",
    "UnivariateSpline",
    "approximate_taylor_polynomial",
    "barycentric_interpolate",
    "bisplev",
    "bisplrep",
    "generate_knots",
    "griddata",
    "insert",
    "interp2d",
    "interpn",
    "krogh_interpolate",
    "lagrange",
    "make_interp_spline",
    "make_lsq_spline",
    "make_smoothing_spline",
    "make_splprep",
    "make_splrep",
    "pade",
    "spalde",
    "splantider",
    "splder",
    "splev",
    "splint",
    "splprep",
    "splrep",
    "sproot",
}


def __getattr__(name):
    if name in _UNSUPPORTED:
        raise NotImplementedError(
            f"scipy.interpolate.{name} is not supported by shellsim's SciPy"
        )
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def _unsupported_method(name):
    raise NotImplementedError(f"{name} is not supported by shellsim's SciPy")


def _float_or_complex(a):
    """`a` as float64, or as complex128 when it holds complex values."""
    a = np.asarray(a)
    return a.astype(np.complex128 if np.iscomplexobj(a) else np.float64)


def _column(a, ndim):
    """1-D `a` reshaped to broadcast along the first axis of an `ndim`-dimensional array."""
    return a.reshape((-1,) + (1,) * (ndim - 1))


def _place_points(values, points_ndim, axis):
    """Move the leading `points_ndim` axes of `values` (the evaluation points) to `axis` among
    the trailing value axes, where SciPy's interpolators put them."""
    if axis == 0:
        return values
    order = (
        list(range(points_ndim, points_ndim + axis))
        + list(range(points_ndim))
        + list(range(points_ndim + axis, values.ndim))
    )
    return values.transpose(order)


# -------------------------------------------------------------------------------------------
# Piecewise polynomials
# -------------------------------------------------------------------------------------------


def _find_intervals(breaks, points):
    """For each point, the index `i` of the piece ``[breaks[i], breaks[i + 1])`` that holds it,
    clamped to the first or last piece outside the breakpoints. Descending breakpoints hold
    ``(breaks[i + 1], breaks[i]]`` instead."""
    if breaks[-1] < breaks[0]:
        breaks, points = -breaks, -points
    found = np.searchsorted(breaks, points, side="right") - 1
    return np.clip(found, 0, len(breaks) - 2)


class PPoly:
    """A piecewise polynomial in the local power basis.

    Between breakpoints ``x[i]`` and ``x[i + 1]`` the value is
    ``sum(c[m, i] * (xp - x[i]) ** (k - m) for m in range(k + 1))``, so ``c`` has shape
    ``(k + 1, len(x) - 1, *trailing)``. Breakpoints may be strictly increasing or strictly
    decreasing. At construction, ``axis`` says where the degree and interval axes sit in the
    given ``c`` (at ``axis`` and ``axis + 1``; the stored ``c`` has them first). In every
    result it places the evaluation points' axes at ``axis`` among the trailing value axes.

    >>> PPoly([[1.0, 2.0], [3.0, 4.0]], [0, 1, 3])(1.5)
    array(5.)
    """

    def __init__(self, c, x, extrapolate=None, axis=0):
        c = _float_or_complex(c)
        x = np.asarray(x, dtype=np.float64)
        axis = operator.index(axis)
        if c.ndim < 2:
            raise ValueError("Coefficients array must be at least 2-dimensional.")
        if not 0 <= axis < c.ndim - 1:
            raise ValueError(f"axis={axis} must be between 0 and {c.ndim - 1}")
        c = np.moveaxis(c, (axis, axis + 1), (0, 1))
        if x.ndim != 1:
            raise ValueError("x must be 1-dimensional")
        if x.size < 2:
            raise ValueError("at least 2 breakpoints are needed")
        if c.shape[0] == 0:
            raise ValueError("polynomial must be at least of order 0")
        if c.shape[1] != x.size - 1:
            raise ValueError("number of coefficients != len(x)-1")
        steps = np.diff(x)
        if not (np.all(steps > 0) or np.all(steps < 0)):
            raise ValueError("`x` must be strictly increasing or decreasing.")
        self.c = c
        self.x = x
        self.axis = axis
        self.extrapolate = _extrapolate_mode(extrapolate)

    @classmethod
    def construct_fast(cls, c, x, extrapolate=None, axis=0):
        """A `cls` holding `c` and `x` as given, which must already have the layout
        `__init__` produces; no validation or copying."""
        poly = object.__new__(cls)
        poly.c = c
        poly.x = x
        poly.axis = axis
        poly.extrapolate = _extrapolate_mode(extrapolate)
        return poly

    def __call__(self, x, nu=0, extrapolate=None):
        """The `nu`-th derivative at `x`, shaped ``trailing[:axis] + x.shape +
        trailing[axis:]``. Points outside the breakpoints extend the end pieces, wrap around
        (``extrapolate="periodic"``) or evaluate to NaN (``extrapolate=False``)."""
        nu = int(nu)
        if nu < 0:
            raise ValueError("Order of derivative cannot be negative")
        if extrapolate is None:
            extrapolate = self.extrapolate
        x = np.asarray(x)
        points = x.astype(np.float64).ravel()
        values = _evaluate(self.c, self.x, points, nu, _extrapolate_mode(extrapolate))
        values = values.reshape(x.shape + self.c.shape[2:])
        return _place_points(values, x.ndim, self.axis)

    def derivative(self, nu=1):
        """The piecewise polynomial of the `nu`-th derivative (an antiderivative for negative
        `nu`), of the same class."""
        if nu < 0:
            return self.antiderivative(-nu)
        k = self.c.shape[0] - 1
        if nu > k:
            c = np.zeros((1,) + self.c.shape[1:], dtype=self.c.dtype)
        else:
            factors = np.array([math.perm(k - m, nu) for m in range(k + 1 - nu)], dtype=float)
            c = self.c[: k + 1 - nu] * _column(factors, self.c.ndim)
        return self.construct_fast(c, self.x, self.extrapolate, self.axis)

    def antiderivative(self, nu=1):
        """The piecewise polynomial of the `nu`-th antiderivative that vanishes at ``x[0]``
        with its first ``nu - 1`` derivatives, continuous across breakpoints. Periodic
        extrapolation does not carry over, since the result is not periodic."""
        if nu <= 0:
            return self.derivative(-nu)
        c = self.c
        widths = np.diff(self.x)
        for _ in range(nu):
            c = _integrate_once(c, widths)
        extrapolate = False if self.extrapolate == "periodic" else self.extrapolate
        return self.construct_fast(c, self.x, extrapolate, self.axis)

    def integrate(self, a, b, extrapolate=None):
        """The definite integral from `a` to `b`, with the trailing value shape. It is NaN
        when the range leaves the breakpoints and `extrapolate` is false."""
        if extrapolate is None:
            extrapolate = self.extrapolate
        extrapolate = _extrapolate_mode(extrapolate)
        a, b, sign = float(a), float(b), 1.0
        if b < a:
            a, b, sign = b, a, -1.0
        pieces = _termwise_integral(self.c)
        lo, hi = min(self.x[0], self.x[-1]), max(self.x[0], self.x[-1])
        if extrapolate == "periodic":
            period = hi - lo
            cycles, rest = divmod(b - a, period)
            start = lo + (a - lo) % period
            end = start + rest
            total = cycles * _integral(pieces, self.x, lo, hi)
            if end <= hi:
                total = total + _integral(pieces, self.x, start, end)
            else:
                total = (
                    total
                    + _integral(pieces, self.x, start, hi)
                    + _integral(pieces, self.x, lo, end - period)
                )
        elif not extrapolate and (a < lo or b > hi):
            total = np.full(self.c.shape[2:], np.nan, dtype=self.c.dtype)
        else:
            total = _integral(pieces, self.x, a, b)
        return np.asarray(sign * total)

    def roots(self, discontinuity=True, extrapolate=None):
        _unsupported_method("PPoly.roots")

    def solve(self, y=0.0, discontinuity=True, extrapolate=None):
        _unsupported_method("PPoly.solve")

    def extend(self, c, x):
        _unsupported_method("PPoly.extend")

    @classmethod
    def from_spline(cls, tck, extrapolate=None):
        _unsupported_method("PPoly.from_spline")

    @classmethod
    def from_bernstein_basis(cls, bp, extrapolate=None):
        _unsupported_method("PPoly.from_bernstein_basis")


def _extrapolate_mode(extrapolate):
    if extrapolate is None:
        return True
    if isinstance(extrapolate, str) and extrapolate == "periodic":
        return extrapolate
    return bool(extrapolate)


def _evaluate(c, breaks, points, nu, extrapolate):
    """The `nu`-th derivative of the pieces `c` at the 1-D float `points`, one row per point."""
    if extrapolate == "periodic":
        points = breaks[0] + (points - breaks[0]) % (breaks[-1] - breaks[0])
        extrapolate = False
    intervals = _find_intervals(breaks, points)
    local = _column(points - breaks[intervals], c.ndim - 1)
    values = _power_sum(c[:, intervals], local, nu)
    if not extrapolate:
        lo, hi = min(breaks[0], breaks[-1]), max(breaks[0], breaks[-1])
        values[(points < lo) | (points > hi)] = np.nan
    return values


def _integral(pieces, breaks, a, b):
    """The integral over ``a <= b`` of the piecewise polynomial whose pieces have the
    term-wise antiderivatives `pieces`: each piece contributes over the part of the range it
    covers, summed from `a` to `b`, which is how SciPy rounds. The end pieces extend beyond
    the breakpoints."""
    first, last = _find_intervals(breaks, np.array([a, b]))
    step = 1 if first <= last else -1
    total = np.zeros(pieces.shape[2:], dtype=pieces.dtype)
    for i in range(first, last + step, step):
        left, right = sorted((breaks[i], breaks[i + 1]))
        lo = a if i == first else left
        hi = b if i == last else right
        total = total + (
            _power_sum(pieces[:, i], hi - breaks[i], 0) - _power_sum(pieces[:, i], lo - breaks[i], 0)
        )
    return total


def _power_sum(c, local, nu):
    """The `nu`-th derivative of the polynomials with coefficients `c` (highest power first)
    at offsets `local`, summed from the constant term up, which is how SciPy rounds."""
    k = c.shape[0] - 1
    values = np.zeros(np.broadcast_shapes(c.shape[1:], np.shape(local)), dtype=c.dtype)
    power = np.ones_like(local)
    for m in range(k - nu, -1, -1):
        values = values + c[m] * power * math.perm(k - m, nu)
        power = power * local
    return values


def _termwise_integral(c):
    """Coefficients of each piece's own antiderivative, with a zero constant term."""
    k = c.shape[0] - 1
    divisors = np.arange(k + 1, 0, -1, dtype=float)
    integrated = np.zeros((k + 2,) + c.shape[1:], dtype=c.dtype)
    integrated[: k + 1] = c / _column(divisors, c.ndim)
    return integrated


def _integrate_once(c, widths):
    """Coefficients of the antiderivative of the pieces `c` whose value at the first
    breakpoint is zero: each piece integrates term by term, and its constant is the value
    the previous piece reaches at the end of its interval, so the result is continuous."""
    k = c.shape[0] - 1
    integrated = _termwise_integral(c)
    for i in range(1, c.shape[1]):
        integrated[k + 1, i] = _power_sum(integrated[:, i - 1], widths[i - 1], 0)
    return integrated


# -------------------------------------------------------------------------------------------
# Cubic splines
# -------------------------------------------------------------------------------------------


def _spline_data(x, y, axis):
    """Validated float breakpoints, the values with the interpolation axis moved first, and
    the normalized axis, with CubicSpline's messages."""
    x = np.asarray(x, dtype=np.float64)
    y = _float_or_complex(y)
    if y.ndim == 0:
        raise ValueError("`y` must be at least 1-dimensional.")
    axis = operator.index(axis) % y.ndim
    if x.ndim != 1:
        raise ValueError("`x` must be 1-dimensional.")
    if x.size < 2:
        raise ValueError("`x` must contain at least 2 elements.")
    if y.shape[axis] != x.size:
        raise ValueError(f"The length of `y` along `axis`={axis} doesn't match the length of `x`")
    if not np.all(np.isfinite(x)):
        raise ValueError("`x` must contain only finite values.")
    if not np.all(np.isfinite(y)):
        raise ValueError("`y` must contain only finite values.")
    if np.any(np.diff(x) <= 0):
        raise ValueError("`x` must be strictly increasing sequence.")
    return x, np.moveaxis(y, axis, 0), axis


def _solve_real_banded(l_and_u, bands, rhs):
    """Solve the real banded system `bands` for each column of the 2-D `rhs`. A complex
    right-hand side is solved as its real and imaginary columns side by side, since the
    matrix itself is real."""
    if not np.iscomplexobj(rhs):
        return solve_banded(l_and_u, bands, rhs, check_finite=False)
    width = rhs.shape[1]
    parts = solve_banded(l_and_u, bands, np.hstack([rhs.real, rhs.imag]), check_finite=False)
    return parts[:, :width] + 1j * parts[:, width:]


def _secants(x, y):
    widths = _column(np.diff(x), y.ndim)
    return widths, np.diff(y, axis=0) / widths


def _hermite_coefficients(x, y, slopes):
    """Power-basis coefficients of the cubic on each interval that matches `y` and `slopes`
    at both ends."""
    widths, secants = _secants(x, y)
    excess = (slopes[:-1] + slopes[1:] - 2 * secants) / widths
    c = np.empty((4, len(x) - 1) + y.shape[1:], dtype=np.result_type(y, slopes))
    c[0] = excess / widths
    c[1] = (secants - slopes[:-1]) / widths - excess
    c[2] = slopes[:-1]
    c[3] = y[:-1]
    return c


class CubicHermiteSpline(PPoly):
    """The piecewise cubic through `y` with first derivatives `dydx` at the points `x`."""

    def __init__(self, x, y, dydx, axis=0, extrapolate=None):
        x, y, axis = _spline_data(x, y, axis)
        dydx = _float_or_complex(dydx)
        if dydx.ndim != y.ndim:
            raise ValueError("The shapes of `y` and `dydx` must be identical.")
        dydx = np.moveaxis(dydx, axis, 0)
        if dydx.shape != y.shape:
            raise ValueError("The shapes of `y` and `dydx` must be identical.")
        PPoly.__init__(self, _hermite_coefficients(x, y, dydx), x, extrapolate)
        self.axis = axis


class CubicSpline(CubicHermiteSpline):
    """The twice continuously differentiable piecewise cubic through `y` at the points `x`.

    `bc_type` fixes the two remaining degrees of freedom: ``"not-a-knot"`` (the default; the
    first two and last two pieces are each one cubic), ``"natural"`` (zero second derivative),
    ``"clamped"`` (zero first derivative), ``"periodic"`` (matching first and second
    derivatives at the ends, which needs ``y[0] == y[-1]``), or a pair of per-end conditions,
    each one of those names or ``(order, value)`` for a first or second derivative value.
    """

    def __init__(self, x, y, axis=0, bc_type="not-a-knot", extrapolate=None):
        x, y, axis = _spline_data(x, y, axis)
        start, end = _boundary_conditions(bc_type, y)
        if extrapolate is None and start == "periodic":
            extrapolate = "periodic"
        if start == "periodic":
            slopes = _periodic_slopes(x, y)
        else:
            slopes = _spline_slopes(x, y, start, end)
        PPoly.__init__(self, _hermite_coefficients(x, y, slopes), x, extrapolate)
        self.axis = axis


_NAMED_CONDITIONS = {"not-a-knot": "not-a-knot", "clamped": (1, 0.0), "natural": (2, 0.0)}


def _boundary_conditions(bc_type, y):
    """The start and end conditions: each ``"not-a-knot"``, ``"periodic"`` or ``(order,
    value)`` with `value` shaped like one sample of `y`."""
    if isinstance(bc_type, str):
        if bc_type == "periodic":
            if not np.allclose(y[0], y[-1], rtol=1e-15, atol=1e-15):
                raise ValueError(
                    "The first and last `y` point along axis 0 must be identical (within "
                    "machine precision) when bc_type='periodic'."
                )
            return "periodic", "periodic"
        bc_type = (bc_type, bc_type)
    else:
        if len(bc_type) != 2:
            raise ValueError(
                "`bc_type` must contain 2 elements to specify start and end conditions."
            )
        if any(isinstance(bc, str) and bc == "periodic" for bc in bc_type):
            raise ValueError(
                "'periodic' `bc_type` is defined for both curve ends and cannot be used with "
                "other boundary conditions."
            )
    return tuple(_boundary_condition(bc, y.shape[1:]) for bc in bc_type)


def _boundary_condition(bc, shape):
    if isinstance(bc, str):
        if bc not in _NAMED_CONDITIONS:
            raise ValueError(f"bc_type={bc} is not allowed.")
        bc = _NAMED_CONDITIONS[bc]
        if bc == "not-a-knot":
            return bc
        return bc[0], np.zeros(shape)
    if len(bc) != 2:
        raise ValueError("A specified derivative value must be given in the form (order, value).")
    order, value = bc
    if order not in (1, 2):
        raise ValueError("The specified derivative order must be 1 or 2.")
    value = np.asarray(value)
    if value.shape != shape:
        raise ValueError(f"`deriv_value` shape {value.shape} is not the expected one {shape}.")
    return order, value


def _spline_slopes(x, y, start, end):
    """The knot slopes of the C2 cubic spline with non-periodic end conditions.

    Continuity of the second derivative at each interior knot gives one tridiagonal equation
    in three neighboring slopes. Each end adds one equation: a given first derivative fixes
    the slope, a given second derivative relates the end slope to its neighbor, and
    not-a-knot (continuity of the third derivative at the second knot) is combined with the
    first interior equation so the system stays tridiagonal.
    """
    n = len(x)
    widths, secants = _secants(x, y)
    h = np.diff(x)
    if n == 2:
        # With a single interval there is no second knot to join across; not-a-knot there
        # means the end slope equals the chord's.
        start = (1, secants[0]) if start == "not-a-knot" else start
        end = (1, secants[0]) if end == "not-a-knot" else end
    elif n == 3 and start == end == "not-a-knot":
        # Both conditions coincide; the spline is the parabola through the three points.
        curvature = (secants[1] - secants[0]) / (x[2] - x[0])
        return np.stack(
            [
                secants[0] - curvature * h[0],
                secants[0] + curvature * h[0],
                secants[1] + curvature * h[1],
            ]
        )
    bands = np.zeros((3, n))
    rhs = np.empty((n,) + y.shape[1:], dtype=secants.dtype)
    bands[0, 2:] = h[:-1]
    bands[1, 1:-1] = 2 * (h[:-1] + h[1:])
    bands[2, :-2] = h[1:]
    rhs[1:-1] = 3 * (widths[1:] * secants[:-1] + widths[:-1] * secants[1:])
    if start == "not-a-knot":
        bands[1, 0] = h[1]
        bands[0, 1] = h[0] + h[1]
        rhs[0] = (
            (3 * h[0] + 2 * h[1]) * h[1] * secants[0] + h[0] ** 2 * secants[1]
        ) / (h[0] + h[1])
    elif start[0] == 1:
        bands[1, 0] = 1.0
        rhs[0] = start[1]
    else:
        bands[1, 0] = 2.0
        bands[0, 1] = 1.0
        rhs[0] = 3 * secants[0] - start[1] * h[0] / 2
    if end == "not-a-knot":
        bands[1, -1] = h[-2]
        bands[2, -2] = h[-1] + h[-2]
        rhs[-1] = (
            (3 * h[-1] + 2 * h[-2]) * h[-2] * secants[-1] + h[-1] ** 2 * secants[-2]
        ) / (h[-1] + h[-2])
    elif end[0] == 1:
        bands[1, -1] = 1.0
        rhs[-1] = end[1]
    else:
        bands[1, -1] = 2.0
        bands[2, -2] = 1.0
        rhs[-1] = 3 * secants[-1] + end[1] * h[-1] / 2
    solution = _solve_real_banded((1, 1), bands, rhs.reshape(n, -1))
    return solution.reshape(rhs.shape)


def _periodic_slopes(x, y):
    """The knot slopes of the periodic C2 cubic spline.

    The slope at the last knot repeats the first, so knot 0's equation wraps around to the
    last interval. That makes the matrix tridiagonal plus two corner entries; the corners are
    folded into a rank-one update solved by the Sherman-Morrison formula, so the solve stays
    banded. Two unknowns or fewer are solved densely, since there the neighbors coincide.
    """
    widths, secants = _secants(x, y)
    h = np.diff(x)
    size = len(x) - 1
    prev = np.roll(np.arange(size), 1)
    rhs = 3 * (widths * secants[prev] + widths[prev] * secants)
    diagonal = 2 * (h[prev] + h)
    if size <= 2:
        matrix = np.diag(diagonal)
        for i in range(size):
            matrix[i, (i - 1) % size] += h[i]
            matrix[i, (i + 1) % size] += h[i - 1]
        slopes = np.linalg.solve(matrix, rhs.reshape(size, -1)).reshape(rhs.shape)
    else:
        top_corner = h[0]  # coefficient of the last slope in knot 0's equation
        bottom_corner = h[-2]  # coefficient of the first slope in the last equation
        gamma = -diagonal[0]
        bands = np.zeros((3, size))
        bands[0, 1:] = h[prev][:-1]
        bands[1] = diagonal
        bands[1, 0] -= gamma
        bands[1, -1] -= bottom_corner * top_corner / gamma
        bands[2, :-1] = h[1:]
        update = np.zeros(size)
        update[0] = gamma
        update[-1] = bottom_corner
        flat = rhs.reshape(size, -1)
        both = _solve_real_banded((1, 1), bands, np.column_stack([flat, update]))
        base, correction = both[:, :-1], both[:, -1]
        dot_base = base[0] + base[-1] * top_corner / gamma
        dot_correction = correction[0] + correction[-1] * top_corner / gamma
        slopes = base - correction[:, None] * (dot_base / (1 + dot_correction))
        slopes = slopes.reshape(rhs.shape)
    return np.concatenate([slopes, slopes[:1]])


class PchipInterpolator(CubicHermiteSpline):
    """The shape-preserving piecewise cubic Hermite interpolant (PCHIP).

    Interior slopes are zero where the data turn and otherwise a weighted harmonic mean of the
    adjacent secant slopes (Fritsch and Carlson's condition for monotonicity); end slopes come
    from a three-point estimate limited to keep the end pieces monotone.
    """

    def __init__(self, x, y, axis=0, extrapolate=None):
        if np.iscomplexobj(np.asarray(y)):
            raise ValueError(
                "`PchipInterpolator` only works with real values for `y`. If you are trying "
                "to use the real components of the passed array, use `np.real` on the array "
                "before passing to `PchipInterpolator`."
            )
        x, y, axis = _spline_data(x, y, axis)
        slopes = _pchip_slopes(x, y)
        PPoly.__init__(self, _hermite_coefficients(x, y, slopes), x, extrapolate)
        self.axis = axis


pchip = PchipInterpolator


def _pchip_slopes(x, y):
    widths, secants = _secants(x, y)
    if len(x) == 2:
        return np.concatenate([secants, secants])
    slopes = np.empty_like(y)
    before, after = widths[:-1], widths[1:]
    weight_before = 2 * after + before
    weight_after = after + 2 * before
    turning = np.sign(secants[:-1]) * np.sign(secants[1:]) <= 0
    with np.errstate(divide="ignore", invalid="ignore"):
        harmonic = 1.0 / (
            (weight_before / secants[:-1] + weight_after / secants[1:])
            / (weight_before + weight_after)
        )
    slopes[1:-1] = np.where(turning, 0.0, harmonic)
    slopes[0] = _pchip_end_slope(widths[0], widths[1], secants[0], secants[1])
    slopes[-1] = _pchip_end_slope(widths[-1], widths[-2], secants[-1], secants[-2])
    return slopes


def _pchip_end_slope(h_end, h_next, secant_end, secant_next):
    """The end slope from the parabola through the three end points, set to zero if it
    points against the end secant, and limited to three times the secant where the data
    turn at the next knot."""
    slope = ((2 * h_end + h_next) * secant_end - h_end * secant_next) / (h_end + h_next)
    slope = np.where(np.sign(slope) != np.sign(secant_end), 0.0, slope)
    overshoot = (np.sign(secant_end) != np.sign(secant_next)) & (
        np.abs(slope) > np.abs(3 * secant_end)
    )
    return np.where(overshoot, 3 * secant_end, slope)


def pchip_interpolate(xi, yi, x, der=0, axis=0):
    """PCHIP interpolation of `(xi, yi)` evaluated at `x`: the `der`-th derivative, or a list
    with one array per order when `der` is a sequence."""
    interpolant = PchipInterpolator(xi, yi, axis=axis)
    if np.ndim(der) == 0:
        return interpolant.derivative(der)(x)
    return [interpolant.derivative(nu)(x) for nu in der]


# -------------------------------------------------------------------------------------------
# interp1d
# -------------------------------------------------------------------------------------------

_SPLINE_KINDS = {"zero": 0, "slinear": 1, "quadratic": 2, "cubic": 3}
_OTHER_KINDS = {"linear", "nearest", "nearest-up", "previous", "next"}


def _broadcastable(shape, target):
    if len(shape) > len(target):
        return False
    return all(have in (1, want) for have, want in zip(reversed(shape), reversed(target)))


class interp1d:
    """Interpolation of a 1-D function sampled at `x`, along `axis` of `y`.

    `kind` is ``"linear"``, ``"nearest"`` (ties go down), ``"nearest-up"`` (ties go up),
    ``"previous"``, ``"next"``, or a spline order given by name (``"zero"``, ``"slinear"``,
    ``"quadratic"``, ``"cubic"``) or as an integer. Spline kinds interpolate with a B-spline
    of that degree whose knots are the interior data points (odd degrees) or their midpoints
    (even degrees), so ``"cubic"`` is the not-a-knot cubic spline.

    Points outside ``[x[0], x[-1]]`` raise ``ValueError`` when `bounds_error` is true (the
    default unless ``fill_value="extrapolate"``). Otherwise they take `fill_value`: one
    array broadcast to the value shape, a ``(below, above)`` pair, or ``"extrapolate"`` to
    extend the end pieces.
    """

    def __init__(self, x, y, kind="linear", axis=-1, copy=True, bounds_error=None,
                 fill_value=np.nan, assume_sorted=False):
        self._order = _spline_order(kind)
        self._kind = "spline" if self._order is not None else kind
        self.copy = copy
        x = np.array(x)
        y = _float_or_complex(y)
        if y.ndim == 0 or x.shape[0] != y.shape[axis]:
            raise ValueError("x and y arrays must be equal in length along interpolation axis.")
        if x.ndim != 1:
            raise ValueError("the x array must have exactly one dimension.")
        if x.size == 0:
            raise ValueError("x and y arrays must have at least 1 entry")
        self.axis = axis % y.ndim
        if not assume_sorted:
            order = np.argsort(x, kind="stable")
            x = x[order]
            y = np.take(y, order, axis=self.axis)
        self.x = x
        self.y = y
        self._values = np.moveaxis(y, self.axis, 0)
        self._set_fill(fill_value, bounds_error)
        if self._order is not None:
            self._prepare_spline()

    def _set_fill(self, fill_value, bounds_error):
        self._extrapolate = isinstance(fill_value, str) and fill_value == "extrapolate"
        if self._extrapolate:
            if bounds_error:
                raise ValueError("Cannot extrapolate and raise at the same time.")
            self.bounds_error = False
            self.fill_value = fill_value
            return
        self.bounds_error = True if bounds_error is None else bounds_error
        target = self._values.shape[1:] or (1,)
        if isinstance(fill_value, tuple) and len(fill_value) == 2:
            below, above = np.asarray(fill_value[0]), np.asarray(fill_value[1])
            for name, fill in (("below", below), ("above", above)):
                if not _broadcastable(fill.shape, target):
                    raise ValueError(
                        f"fill_value ({name}) argument must be able to broadcast up to shape "
                        f"{target} but had shape {fill.shape}"
                    )
            self.fill_value = fill_value
        else:
            below = above = np.asarray(fill_value)
            if not _broadcastable(below.shape, target):
                raise ValueError(
                    f"fill_value argument must be able to broadcast up to shape {target} but "
                    f"had shape {below.shape}"
                )
            self.fill_value = below
        self._fill_below, self._fill_above = below, above

    def _prepare_spline(self):
        order = self._order
        n = len(self.x)
        if np.any(np.diff(self.x) == 0):
            raise ValueError("Expect x to not have duplicates")
        if order == 1 and n < 2:
            raise ValueError("x and y arrays must have at least 2 entries")
        if order >= 1:
            if n < order + 1:
                raise ValueError(
                    "The number of derivatives at boundaries does not match: expected "
                    f"{order + 1 - n}, got 0+0"
                )
            self._knots, self._coefficients = _interpolating_bspline(
                self.x.astype(np.float64), self._values, order
            )

    def __call__(self, x):
        """The interpolated values at `x`, shaped with `x`'s shape at `axis` of `y`'s."""
        x = np.asarray(x)
        points = x.astype(np.float64).ravel()
        values = self._evaluate(points)
        if not self._extrapolate:
            below = points < self.x[0]
            above = points > self.x[-1]
            if self.bounds_error and below.any():
                raise ValueError(
                    f"A value ({points[np.argmax(below)]}) in x_new is below the "
                    f"interpolation range's minimum value ({self.x[0]})."
                )
            if self.bounds_error and above.any():
                raise ValueError(
                    f"A value ({points[np.argmax(above)]}) in x_new is above the "
                    f"interpolation range's maximum value ({self.x[-1]})."
                )
            values[below] = self._fill_below
            values[above] = self._fill_above
        values = values.reshape(x.shape + self._values.shape[1:])
        return _place_points(values, x.ndim, self.axis)

    def _evaluate(self, points):
        kind, x, y = self._kind, self.x, self._values
        n = len(x)
        if kind == "spline" and self._order >= 1:
            return _bspline_values(self._knots, self._coefficients, self._order, points)
        if kind == "linear":
            return _linear(x, y, points, self._extrapolate)
        if kind in ("nearest", "nearest-up"):
            midpoints = (x[:-1] + x[1:]) / 2
            side = "left" if kind == "nearest" else "right"
            return y[np.clip(np.searchsorted(midpoints, points, side=side), 0, n - 1)]
        if kind == "previous" or kind == "spline":
            found = np.searchsorted(x, points, side="right") - 1
            values = y[np.clip(found, 0, n - 1)]
            if kind == "previous":
                values[found < 0] = np.nan
            else:
                values[np.isnan(points)] = np.nan
            return values
        found = np.searchsorted(x, points, side="left")
        values = y[np.clip(found, 0, n - 1)]
        values[found >= n] = np.nan
        return values


def _spline_order(kind):
    if isinstance(kind, str):
        if kind in _SPLINE_KINDS:
            return _SPLINE_KINDS[kind]
        if kind in _OTHER_KINDS:
            return None
        raise NotImplementedError(f"{kind} is unsupported: Use fitpack routines for other types.")
    order = operator.index(kind)
    if order < 0:
        raise ValueError("Expect non-negative k.")
    return order


def _linear(x, y, points, extrapolate):
    """Linear interpolation between the samples around each point. Like SciPy, 1-D real
    samples read inside the range go through ``np.interp`` (a point on a repeated abscissa
    takes the last sample there); otherwise the two neighbors are weighted by distance, which
    rounds differently in the last bit."""
    if y.ndim == 1 and not np.iscomplexobj(y) and not extrapolate:
        return np.interp(points, x, y)
    n = len(x)
    if n == 1:
        values = y[np.zeros(len(points), dtype=np.intp)]
        values[points != x[0]] = np.nan
        return values
    lo = np.clip(np.searchsorted(x, points, side="right") - 1, 0, n - 2)
    hi = lo + 1
    x_lo, x_hi = x[lo], x[hi]
    span = x_hi - x_lo
    weight_hi = _column((points - x_lo) / span, y.ndim)
    weight_lo = _column((x_hi - points) / span, y.ndim)
    return weight_hi * y[hi] + weight_lo * y[lo]


def _interpolating_bspline(x, y, order):
    """Knots and coefficients of the degree-`order` B-spline through `(x, y)`, with the
    boundary knots repeated ``order + 1`` times and the interior knots at the data points
    (odd degree) or at the midpoints between them (even degree), so there are as many basis
    functions as points. The collocation matrix is banded, and so is its solve."""
    n = len(x)
    if order % 2:
        half = (order + 1) // 2
        interior = x[half : n - half]
    else:
        half = order // 2
        interior = (x[half : n - half - 1] + x[half + 1 : n - half]) / 2
    knots = np.concatenate([np.full(order + 1, x[0]), interior, np.full(order + 1, x[-1])])
    if order == 1:
        # Degree-1 B-splines are hat functions peaking at the data points.
        return knots, y
    intervals = _knot_intervals(knots, order, n, x)
    basis = _bspline_basis(knots, order, x, intervals)
    rows = np.arange(n)
    first = intervals - order
    lower = int(np.max(rows - first))
    upper = int(np.max(first + order - rows))
    bands = np.zeros((lower + upper + 1, n))
    for r in range(order + 1):
        bands[upper + rows - first - r, first + r] = basis[:, r]
    flat = y.reshape(n, -1)
    coefficients = _solve_real_banded((lower, upper), bands, flat)
    return knots, coefficients.reshape(y.shape)


def _knot_intervals(knots, order, count, points):
    """The knot interval ``[knots[l], knots[l + 1])`` of each point, clamped to the
    ``order <= l < count`` range where the spline is defined."""
    found = np.searchsorted(knots, points, side="right") - 1
    return np.clip(found, order, count - 1)


def _bspline_basis(knots, order, points, intervals):
    """The ``order + 1`` B-splines of degree `order` that can be nonzero at each point,
    evaluated there by the Cox-de Boor recurrence: column ``r`` is the basis function
    starting at knot ``l - order + r`` for the point's interval ``l``."""
    basis = np.zeros((len(points), order + 1))
    basis[:, 0] = 1.0
    left = [None]
    right = [None]
    for degree in range(1, order + 1):
        left.append(points - knots[intervals + 1 - degree])
        right.append(knots[intervals + degree] - points)
        carried = np.zeros(len(points))
        for r in range(degree):
            share = basis[:, r] / (right[r + 1] + left[degree - r])
            basis[:, r] = carried + right[r + 1] * share
            carried = left[degree - r] * share
        basis[:, degree] = carried
    return basis


def _bspline_values(knots, coefficients, order, points):
    """The B-spline's values at `points`, one row per point; points outside the data range
    extend the end pieces."""
    intervals = _knot_intervals(knots, order, len(coefficients), points)
    basis = _bspline_basis(knots, order, points, intervals)
    values = np.zeros((len(points),) + coefficients.shape[1:], dtype=coefficients.dtype)
    for r in range(order + 1):
        values = values + _column(basis[:, r], values.ndim) * coefficients[intervals - order + r]
    return values
