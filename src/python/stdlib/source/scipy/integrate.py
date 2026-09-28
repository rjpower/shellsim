"""shellsim's ``scipy.integrate``: quadrature of samples, adaptive quadrature of callables, and
initial-value ODE solvers.

Implemented: ``trapezoid``, ``cumulative_trapezoid`` and ``simpson`` over sampled data; the
adaptive ``quad`` (finite or infinite limits) with the thin wrapper ``dblquad``; and
``solve_ivp``/``odeint``, both built on one adaptive Dormand-Prince RK45 stepper. ``nquad``,
``tplquad``, ``quad``'s ``full_output``/``points``/``weight`` and ODE methods other than RK45
are not provided; results agree with SciPy within the reported or requested tolerance, not
bit for bit.
"""

import math
import warnings

import numpy as np

__all__ = [
    "IntegrationWarning",
    "cumulative_trapezoid",
    "dblquad",
    "odeint",
    "quad",
    "simpson",
    "solve_ivp",
    "trapezoid",
]


# -------------------------------------------------------------------------------------------
# quad, dblquad
# -------------------------------------------------------------------------------------------


class IntegrationWarning(UserWarning):
    """Warning that ``quad`` could not reach the requested tolerance."""


_EPSILON = 2.220446049250313e-16
_TINY = 2.2250738585072014e-308
# Extrapolation reads only the latest totals, bounding the work of each refinement step.
_WYNN_TERMS = 50


class _Rule:
    """A Gauss-Kronrod quadrature pair on ``[-1, 1]``, from Piessens et al.'s published node
    and weight tables (QUADPACK, 1983).

    ``nodes`` holds the positive Kronrod abscissae, paired with their ``weights``; ``center``
    is the Kronrod weight at zero. The embedded Gauss rule reuses every other Kronrod node
    (``nodes[1::2]``), weighted by ``gauss_weights`` (and ``gauss_center`` at zero, when zero is
    also a Gauss node); the difference between the two estimates sizes the error.
    """

    def __init__(self, nodes, weights, center, gauss_weights, gauss_center):
        self.nodes = nodes
        self.weights = weights
        self.center = center
        self.gauss_weights = gauss_weights
        self.gauss_center = gauss_center

    def apply(self, f, lo, hi):
        """Return ``(integral, abserr)`` of the rule over ``[lo, hi]``."""
        mid = 0.5 * (lo + hi)
        half = 0.5 * (hi - lo)
        f_mid = f(mid)
        samples = [(self.center, f_mid)]
        kronrod = self.center * f_mid
        gauss = self.gauss_center * f_mid
        for index, node in enumerate(self.nodes):
            offset = half * node
            left, right = f(mid - offset), f(mid + offset)
            weight = self.weights[index]
            samples.append((weight, left))
            samples.append((weight, right))
            kronrod += weight * (left + right)
            if index % 2 == 1:
                gauss += self.gauss_weights[index // 2] * (left + right)
        mean = 0.5 * kronrod
        spread = sum(weight * abs(value - mean) for weight, value in samples) * abs(half)
        error = abs((kronrod - gauss) * half)
        # QUADPACK's error estimate: the Gauss-Kronrod difference scaled by how smooth the
        # integrand looks nearby, and never below what rounding in the sums can resolve.
        if spread != 0 and error != 0:
            error = spread * min(1.0, (200.0 * error / spread) ** 1.5)
        error = max(50.0 * _EPSILON * abs(kronrod * half), error)
        return kronrod * half, error


# The 21-point rule integrates a finite piece directly; the 15-point rule integrates the image
# of an infinite piece under the ``(0, 1]`` map in ``_infinite_integrand``, where the mapped
# integrand typically has more curvature near ``0``, so SciPy's QAGI also uses the smaller pair.
_KRONROD21 = _Rule(
    nodes=(
        0.995657163025808080735527280689003,
        0.973906528517171720077964012084452,
        0.930157491355708226001207180059508,
        0.865063366688984510732096688423493,
        0.780817726586416897063717578345042,
        0.679409568299024406234327365114874,
        0.562757134668604683339000099272694,
        0.433395394129247190799265943165784,
        0.294392862701460198131126603103866,
        0.148874338981631210884826001129720,
    ),
    weights=(
        0.011694638867371874278064396062192,
        0.032558162307964727478818972459390,
        0.054755896574351996031381300244580,
        0.075039674810919952767043140916190,
        0.093125454583697605535065465083366,
        0.109387158802297641899210590325805,
        0.123491976262065851077208980053800,
        0.134709217311473325928054001771707,
        0.142775938577060080797094273138717,
        0.147739104901338491374841515972068,
    ),
    center=0.149445554002916905664936468389821,
    gauss_weights=(
        0.066671344308688137593568809893332,
        0.149451349150580593145776339657697,
        0.219086362515982043995534934228163,
        0.269266719309996355091226921569469,
        0.295524224714752870173892994651338,
    ),
    gauss_center=0.0,
)

_KRONROD15 = _Rule(
    nodes=(
        0.991455371120812639206854697526329,
        0.949107912342758524526189684047851,
        0.864864423359769072789712788640926,
        0.741531185599394439863864773280788,
        0.586087235467691130294144845693013,
        0.405845151377397166906606412076961,
        0.207784955007898467600689403773245,
    ),
    weights=(
        0.022935322010529224963732008058970,
        0.063092092629978553290700663189204,
        0.104790010322250183839876322541518,
        0.140653259715525918745189590510238,
        0.169004726639267902826583426598550,
        0.190350578064785409913256402421014,
        0.204432940075298892414161999234649,
    ),
    center=0.209482141084727828012999174891714,
    gauss_weights=(
        0.129484966168869693270611432679082,
        0.279705391489276667901467771423780,
        0.381830050505118944950369775488975,
    ),
    gauss_center=0.417959183673469387755102040816327,
)


def _real(value):
    """Convert an integrand's return value to ``float``, as SciPy's compiled wrapper does."""
    if isinstance(value, (str, bytes, list, tuple, dict, complex)) or value is None:
        raise TypeError(f"must be real number, not {type(value).__name__}")
    return float(value)


def _wynn(sequence):
    """The last even-column entry of Wynn's epsilon table for ``sequence``.

    The even columns hold Shanks-transform estimates of the sequence's limit. Construction
    stops early when two neighbouring entries coincide, since the next column would divide by
    zero.
    """
    previous = [0.0] * (len(sequence) + 1)
    current = list(sequence)
    best = current[-1]
    column = 0
    while len(current) > 1:
        following = []
        for k in range(len(current) - 1):
            delta = current[k + 1] - current[k]
            if delta == 0:
                return best
            following.append(previous[k + 1] + 1.0 / delta)
        previous, current = current, following
        column += 1
        if column % 2 == 0:
            best = current[-1]
    return best


class _Piece:
    """One piece of the integration range with its Gauss-Kronrod estimates."""

    def __init__(self, lo, hi, area, error):
        self.lo = lo
        self.hi = hi
        self.area = area
        self.error = error


class _Partition:
    """The pieces of the integration range, refined by repeatedly bisecting the worst one."""

    def __init__(self, f, rule, bounds, limit):
        self.f = f
        self.rule = rule
        self.limit = limit
        self.pieces = [self._piece(lo, hi) for lo, hi in zip(bounds, bounds[1:])]

    def _piece(self, lo, hi):
        area, error = self.rule.apply(self.f, lo, hi)
        return _Piece(lo, hi, area, error)

    def area(self):
        return sum(piece.area for piece in self.pieces)

    def error(self):
        return sum(piece.error for piece in self.pieces)

    def worst(self):
        return max(range(len(self.pieces)), key=lambda index: self.pieces[index].error)

    def bisect(self, index):
        """Split the piece at ``index`` in two. False when ``limit`` pieces already exist or
        the piece is already as small as floating point allows, either of which ends
        refinement."""
        if len(self.pieces) >= self.limit:
            return False
        piece = self.pieces[index]
        mid = 0.5 * (piece.lo + piece.hi)
        if not min(piece.lo, piece.hi) < mid < max(piece.lo, piece.hi):
            return False
        self.pieces[index] = self._piece(piece.lo, mid)
        self.pieces.append(self._piece(mid, piece.hi))
        return True


def _adapt(partition, epsabs, epsrel):
    """Refine ``partition``, bisecting its worst-error piece, until the total meets
    ``max(epsabs, epsrel * |result|)``. Returns ``(result, abserr, converged)``.

    The running totals after each bisection also feed Wynn's epsilon algorithm. Near an
    integrable singularity the raw Gauss-Kronrod error estimate can stall (bisecting an
    ever-smaller interval around the singular point keeps a similar relative error) while the
    sequence of totals still converges; epsilon acceleration reads off that limit early, the
    way QUADPACK's QAGS uses it.
    """

    def tolerance(value):
        return max(epsabs, epsrel * abs(value))

    def plausible(estimate, raw_area):
        """Whether an extrapolated `estimate` still agrees with the un-extrapolated running
        sum `raw_area`, guarding against Wynn acceleration finding a spurious limit for a
        divergent integral (which QUADPACK's QAGS also flags this way)."""
        if raw_area == 0:
            return abs(estimate) <= tolerance(estimate)
        ratio = estimate / raw_area
        return (estimate > 0) == (raw_area > 0) and 0.01 <= abs(ratio) <= 100.0

    area, error = partition.area(), partition.error()
    totals, estimates, best = [area], [], None
    while error > tolerance(area) and math.isfinite(area + error):
        if not partition.bisect(partition.worst()):
            break
        area, error = partition.area(), partition.error()
        totals.append(area)
        if len(totals) < 3:
            continue
        estimates.append(_wynn(totals[-_WYNN_TERMS:]))
        if len(estimates) < 2:
            continue
        spread = max(abs(estimates[-1] - estimates[-2]), 4.0 * _EPSILON * abs(estimates[-1]))
        if best is None or spread < best[1]:
            best = (estimates[-1], spread)
        if spread <= tolerance(estimates[-1]):
            return estimates[-1], spread, plausible(estimates[-1], area)
    converged = math.isfinite(area + error) and error <= tolerance(area)
    if not converged and best is not None and best[1] < error:
        return best[0], best[1], best[1] <= tolerance(best[0]) and plausible(best[0], area)
    return area, error, converged


def _infinite_integrand(f, a, b):
    """Map an infinite range onto ``(0, 1]`` with ``x = a + (1 - t) / t`` (or the symmetric
    map when both bounds are infinite)."""
    if a == -math.inf and b == math.inf:
        return lambda t: (f((1.0 - t) / t) + f(-(1.0 - t) / t)) / t / t
    if b == math.inf:
        return lambda t: f(a + (1.0 - t) / t) / t / t
    return lambda t: f(b - (1.0 - t) / t) / t / t


def quad(func, a, b, args=(), epsabs=1.49e-8, epsrel=1.49e-8, limit=50):
    """Integrate ``func(x, *args)`` from ``a`` to ``b``; either bound may be infinite.

    Returns ``(result, abserr)``. When the estimated error cannot be brought under
    ``max(epsabs, epsrel * |result|)`` within ``limit`` subdivisions, an ``IntegrationWarning``
    is raised alongside the best estimate found.

    >>> quad(lambda x: x * x, 0, 1)
    (0.3333333333333333, 3.700743415417188e-15)
    """
    if not isinstance(args, tuple):
        args = (args,)
    if not callable(func):
        raise ValueError("invalid callable given")
    if limit < 1:
        raise ValueError("`limit` must be at least 1")
    flip = b < a
    if flip:
        a, b = b, a
    a, b = float(a), float(b)

    def f(x):
        return _real(func(x, *args))

    if a == b:
        result, error, converged = 0.0, 0.0, True
    elif math.isinf(a) or math.isinf(b):
        partition = _Partition(_infinite_integrand(f, a, b), _KRONROD15, (0.0, 1.0), limit)
        result, error, converged = _adapt(partition, epsabs, epsrel)
    else:
        partition = _Partition(f, _KRONROD21, (a, b), limit)
        result, error, converged = _adapt(partition, epsabs, epsrel)
    if flip:
        result = -result
    if not converged:
        warnings.warn(
            "the integral did not converge to the requested tolerance within the "
            "subdivision limit",
            IntegrationWarning,
            stacklevel=2,
        )
    return result, error


def dblquad(func, a, b, gfun, hfun, args=(), epsabs=1.49e-8, epsrel=1.49e-8):
    """Integrate ``func(y, x, *args)`` for ``x`` in ``[a, b]`` and ``y`` from ``gfun(x)`` to
    ``hfun(x)``; the inner bounds may also be constants.

    >>> dblquad(lambda y, x: x * y, 0, 1, 0, lambda x: x)[0]
    0.125
    """

    def inner(x):
        lo = gfun(x) if callable(gfun) else gfun
        hi = hfun(x) if callable(hfun) else hfun
        return quad(lambda y: func(y, x, *args), lo, hi, epsabs=epsabs, epsrel=epsrel)[0]

    return quad(inner, a, b, epsabs=epsabs, epsrel=epsrel)


# -------------------------------------------------------------------------------------------
# Integration of sampled data
# -------------------------------------------------------------------------------------------

trapezoid = np.trapezoid


def _along(ndim, axis, index):
    """A tuple index that applies ``index`` to one axis and takes every other axis whole."""
    key = [slice(None)] * ndim
    key[axis] = index
    return tuple(key)


def _spacing(y, x, axis):
    """The differences of ``x`` along ``axis``, shaped to broadcast against ``y``."""
    x = np.asarray(x)
    if x.ndim == 1:
        shape = [1] * y.ndim
        shape[axis] = -1
        return np.diff(x).reshape(shape)
    if x.ndim != y.ndim:
        raise ValueError("If given, shape of x must be 1-D or the same as y.")
    return np.diff(x, axis=axis)


def cumulative_trapezoid(y, x=None, dx=1.0, axis=-1, initial=None):
    """Running trapezoid-rule integrals of ``y`` along ``axis``.

    The result has one fewer sample along ``axis`` than ``y``, or the same number when
    ``initial=0`` prepends a zero.

    >>> cumulative_trapezoid([1, 2, 3], initial=0).tolist()
    [0.0, 1.5, 4.0]
    """
    y = np.asarray(y)
    if y.shape[axis] == 0:
        raise ValueError("At least one point is required along `axis`.")
    if x is None:
        d = dx
    else:
        d = _spacing(y, x, axis)
        if d.shape[axis] != y.shape[axis] - 1:
            raise ValueError("If given, length of x along axis must be the same as y.")
    upper = y[_along(y.ndim, axis, slice(1, None))]
    lower = y[_along(y.ndim, axis, slice(None, -1))]
    result = np.cumsum(d * (upper + lower) / 2.0, axis=axis)
    if initial is None:
        return result
    if initial != 0:
        raise ValueError("`initial` must be `None` or `0`.")
    shape = list(result.shape)
    shape[axis] = 1
    return np.concatenate([np.zeros(shape, dtype=result.dtype), result], axis=axis)


def _divide(numerator, denominator):
    """``numerator / denominator``, or zero where the denominator is zero, without a warning.

    Coincident sample points give zero-width intervals, which contribute nothing.
    """
    if np.ndim(denominator) == 0:
        return numerator / denominator if denominator != 0 else 0 * numerator
    with np.errstate(divide="ignore", invalid="ignore"):
        quotient = numerator / denominator
    return np.where(denominator != 0, quotient, 0.0)


def _simpson_pairs(y, h, axis):
    """Simpson's rule over an odd number of samples, one parabola per pair of intervals.

    ``h`` is the uniform spacing, or the interval widths along ``axis`` for uneven samples.
    """
    ndim = y.ndim
    y0 = y[_along(ndim, axis, slice(0, -2, 2))]
    y1 = y[_along(ndim, axis, slice(1, -1, 2))]
    y2 = y[_along(ndim, axis, slice(2, None, 2))]
    if np.ndim(h) == 0:
        return h / 3.0 * np.sum(y0 + 4.0 * y1 + y2, axis)
    h0 = h[_along(ndim, axis, slice(0, None, 2))]
    h1 = h[_along(ndim, axis, slice(1, None, 2))]
    h_sum = h0 + h1
    ratio = _divide(h0, h1)
    weighted = (
        y0 * (2.0 - _divide(1.0, ratio))
        + y1 * (h_sum * _divide(h_sum, h0 * h1))
        + y2 * (2.0 - ratio)
    )
    return np.sum(h_sum / 6.0 * weighted, axis)


def simpson(y, x=None, *, dx=1.0, axis=-1):
    """Integrate samples ``y`` along ``axis`` with composite Simpson's rule.

    An odd number of samples is covered by parabolas through consecutive triples. For an even
    number, the parabola through the last three samples also covers the final interval, as in
    Cartwright's correction. Two samples fall back to the trapezoid rule and one gives zero.

    >>> float(simpson([1, 2, 3, 4]))
    7.5
    """
    y = np.asarray(y)
    n = y.shape[axis]
    if x is not None:
        x = np.asarray(x)
        if x.ndim not in (1, y.ndim):
            raise ValueError("If given, shape of x must be 1-D or the same as y.")
        if x.shape[axis if x.ndim > 1 else 0] != n:
            raise ValueError("If given, length of x along axis must be the same as y.")
    if n == 0:
        raise IndexError("cannot integrate an empty axis")
    if n < 3:
        return trapezoid(y, x, dx=dx, axis=axis)
    h = dx if x is None else _spacing(y, x, axis)
    if n % 2 == 1:
        return _simpson_pairs(y, h, axis)
    head = _along(y.ndim, axis, slice(None, -1))
    if np.ndim(h) == 0:
        result = _simpson_pairs(y[head], h, axis)
        h0 = h1 = h
    else:
        result = _simpson_pairs(y[head], h[head], axis)
        h0 = h[_along(y.ndim, axis, -2)]
        h1 = h[_along(y.ndim, axis, -1)]
    alpha = _divide(2 * h1**2 + 3 * h0 * h1, 6 * (h0 + h1))
    beta = _divide(h1**2 + 3 * h0 * h1, 6 * h0)
    eta = _divide(h1**3, 6 * h0 * (h0 + h1))
    last = y[_along(y.ndim, axis, -1)]
    middle = y[_along(y.ndim, axis, -2)]
    first = y[_along(y.ndim, axis, -3)]
    return result + (alpha * last + beta * middle - eta * first)


# -------------------------------------------------------------------------------------------
# solve_ivp, odeint
# -------------------------------------------------------------------------------------------

# The Dormand-Prince 5(4) Butcher tableau (Dormand & Prince, 1980): the "RK45" adaptive-step
# method underlying SciPy's default solver, reimplemented here from the published coefficients.
# Row 7 repeats the 5th-order weights `_B`, so stage 7 is `fun` at the *next* accepted point
# (the "first same as last" property `_rk45_integrate` exploits to save an evaluation per step).
_C = (0.0, 1 / 5, 3 / 10, 4 / 5, 8 / 9, 1.0, 1.0)
_A = (
    (),
    (1 / 5,),
    (3 / 40, 9 / 40),
    (44 / 45, -56 / 15, 32 / 9),
    (19372 / 6561, -25360 / 2187, 64448 / 6561, -212 / 729),
    (9017 / 3168, -355 / 33, 46732 / 5247, 49 / 176, -5103 / 18656),
    (35 / 384, 0.0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84),
)
_B = (35 / 384, 0.0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84, 0.0)
_B_STAR = (5179 / 57600, 0.0, 7571 / 16695, 393 / 640, -92097 / 339200, 187 / 2100, 1 / 40)


def _rk45_step(fun, t, y, h, f0):
    """One Dormand-Prince step of size ``h`` from ``(t, y)``, given ``f0 = fun(t, y)``.

    Returns the 5th-order estimate, the difference from the embedded 4th-order estimate (used
    to size the local error), and the 7 stage derivatives (the last is ``fun`` at the new
    point, by construction of the tableau).
    """
    stages = [f0]
    for row in _A[1:]:
        stage_t = t + _C[len(stages)] * h
        stage_y = y + h * sum(a * k for a, k in zip(row, stages))
        stages.append(fun(stage_t, stage_y))
    y_next = y + h * sum(b * k for b, k in zip(_B, stages))
    error = h * sum((b - bstar) * k for b, bstar, k in zip(_B, _B_STAR, stages))
    return y_next, error, stages


def _error_norm(error, y0, y1, atol, rtol):
    """The RMS norm of ``error`` scaled by ``atol + rtol * max(|y0|, |y1|)``; at most 1 means
    the step meets the tolerance."""
    scale = atol + rtol * np.maximum(np.abs(y0), np.abs(y1))
    return math.sqrt(np.mean((error / scale) ** 2))


def _rk45_integrate(fun, t0, tf, y0, rtol, atol, max_step):
    """Integrate from ``t0`` to ``tf`` with adaptive Dormand-Prince steps.

    Returns the accepted step times, states and derivatives (each a list, for the dense cubic
    Hermite output ``_dense_evaluate`` builds from them), the number of `fun` evaluations, and
    whether the solver reached ``tf``.
    """
    direction = 1.0 if tf >= t0 else -1.0
    span = abs(tf - t0)
    t, y, f = t0, y0, fun(t0, y0)
    nodes_t, nodes_y, nodes_f = [t], [y], [f]
    nfev = 1
    if span == 0.0:
        return nodes_t, nodes_y, nodes_f, nfev, True
    h = direction * min(max_step, span)
    safety, min_factor, max_factor = 0.9, 0.2, 5.0
    tiny_step = np.finfo(float).eps * 16
    while (tf - t) * direction > 0:
        if abs(h) > abs(tf - t):
            h = tf - t
        y_next, error, stages = _rk45_step(fun, t, y, h, f)
        nfev += 6
        norm = _error_norm(error, y, y_next, atol, rtol)
        if norm <= 1.0:
            t, y, f = t + h, y_next, stages[-1]
            nodes_t.append(t)
            nodes_y.append(y)
            nodes_f.append(f)
            factor = max_factor if norm == 0.0 else min(max_factor, safety * norm**-0.2)
        else:
            factor = max(min_factor, safety * norm**-0.2)
        h = direction * min(abs(h) * factor, max_step)
        if abs(h) < tiny_step * max(abs(t), 1.0):
            return nodes_t, nodes_y, nodes_f, nfev, False
    return nodes_t, nodes_y, nodes_f, nfev, True


def _dense_evaluate(nodes_t, nodes_y, nodes_f, t_eval):
    """Cubic Hermite interpolation of the accepted steps at ``t_eval``, matching each step's
    value and derivative at both ends; shaped ``(len(y0), len(t_eval))``."""
    order = np.argsort(nodes_t)
    ts, ys, fs = nodes_t[order], nodes_y[order], nodes_f[order]
    query = np.asarray(t_eval, dtype=float)
    index = np.clip(np.searchsorted(ts, query, side="right") - 1, 0, len(ts) - 2)
    t0, t1 = ts[index], ts[index + 1]
    y0, y1 = ys[index], ys[index + 1]
    f0, f1 = fs[index], fs[index + 1]
    h = (t1 - t0)[:, None]
    s = ((query - t0) / (t1 - t0))[:, None]
    h00 = (1 + 2 * s) * (1 - s) ** 2
    h10 = s * (1 - s) ** 2
    h01 = s**2 * (3 - 2 * s)
    h11 = s**2 * (s - 1)
    return (h00 * y0 + h10 * h * f0 + h01 * y1 + h11 * h * f1).T


class OdeResult:
    """The result of `solve_ivp`/`odeint`: `t`, `y` (shape ``(len(y0), len(t))``), `success`,
    `status` (0 on success, -1 otherwise), `message` and `nfev`."""

    def __init__(self, t, y, success, status, message, nfev):
        self.t = t
        self.y = y
        self.success = success
        self.status = status
        self.message = message
        self.nfev = nfev


def solve_ivp(
    fun, t_span, y0, method="RK45", t_eval=None, args=None, rtol=1e-3, atol=1e-6, max_step=math.inf
):
    """Integrate the initial-value problem ``dy/dt = fun(t, y)`` from ``t_span[0]`` to
    ``t_span[1]`` starting at ``y0``, with adaptive Dormand-Prince RK45 step control. Only
    ``method="RK45"`` is supported.

    Without ``t_eval`` the returned ``t`` is the solver's own accepted steps; with it, ``y`` is
    cubic-Hermite-interpolated from the step values and derivatives at the requested times.

    >>> solve_ivp(lambda t, y: -y, (0, 1), [1.0], t_eval=[0, 1]).y[0]
    array([1.        , 0.36804663])
    """
    if method != "RK45":
        raise NotImplementedError(
            f"solve_ivp(method={method!r}) is not supported by shellsim's SciPy"
        )
    args = () if args is None else tuple(args)
    t0, tf = float(t_span[0]), float(t_span[1])
    y0 = np.atleast_1d(np.asarray(y0, dtype=float))

    def f(t, y):
        return np.asarray(fun(t, y, *args), dtype=float)

    nodes_t, nodes_y, nodes_f, nfev, success = _rk45_integrate(f, t0, tf, y0, rtol, atol, max_step)
    nodes_t, nodes_y, nodes_f = np.array(nodes_t), np.array(nodes_y), np.array(nodes_f)
    if t_eval is None:
        t_out, y_out = nodes_t, nodes_y.T
    elif len(nodes_t) == 1:
        t_out = np.asarray(t_eval, dtype=float)
        y_out = np.tile(nodes_y[0][:, None], (1, t_out.size))
    else:
        t_out = np.asarray(t_eval, dtype=float)
        y_out = _dense_evaluate(nodes_t, nodes_y, nodes_f, t_out)
    message = (
        "the solver successfully reached the end of the integration interval"
        if success
        else "the required step size became smaller than floating-point spacing"
    )
    return OdeResult(t_out, y_out, success, 0 if success else -1, message, nfev)


def odeint(func, y0, t, args=()):
    """A thin wrapper over the same RK45 integrator as `solve_ivp`, in the calling convention
    of SciPy's older `odeint`: `func(y, t, *args)` and a result shaped ``(len(t), len(y0))``.

    >>> odeint(lambda y, t: -y, 1.0, [0, 1])[:, 0]
    array([1.        , 0.36787944])
    """
    t = np.asarray(t, dtype=float)
    y0 = np.atleast_1d(np.asarray(y0, dtype=float))
    result = solve_ivp(
        lambda s, y: func(y, s, *args),
        (t[0], t[-1]),
        y0,
        t_eval=t,
        rtol=1e-8,
        atol=1e-10,
    )
    return result.y.T
