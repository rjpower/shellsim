"""Adaptive quadrature of Python callables: ``quad``, ``dblquad``, ``tplquad`` and ``nquad``.

``quad`` follows the strategy of Piessens et al.'s QUADPACK (1983), reimplemented from its
published description. Each piece of the range is integrated with a Gauss-Kronrod pair: the
21-point rule on finite pieces, or the 15-point rule on ``(0, 1]`` after mapping an infinite
range there with ``x = a + (1 - t) / t``. The piece with the largest error estimate is bisected
until the summed estimate meets ``max(epsabs, epsrel * |result|)``. When the bisections close in
on a point, as at an integrable singularity, the running totals are extrapolated with Wynn's
epsilon algorithm. Results agree with SciPy's within the reported error, not bit for bit.

The integrand is called with Python floats, in the same order as SciPy's, and must return a
real number. Weighted integrals (``weight=``) are not supported.
"""

import math
import warnings

import numpy as np

__all__ = ["IntegrationWarning", "dblquad", "nquad", "quad", "tplquad"]


class IntegrationWarning(UserWarning):
    """Warning on issues during integration."""


_EPSILON = 2.220446049250313e-16
_TINY = 2.2250738585072014e-308
# Extrapolation uses at most this many of the latest totals, bounding each step's work.
_WYNN_TERMS = 50


class _Rule:
    """A Gauss-Kronrod pair on ``[-1, 1]``.

    ``nodes`` are the positive Kronrod abscissae in decreasing order, with ``weights`` their
    Kronrod weights; ``center`` is the weight at zero. The Gauss rule reuses every other node,
    ``nodes[1::2]``, with ``gauss_weights`` and ``gauss_center`` (zero when zero is not a Gauss
    node). ``order`` lists node indices in evaluation order.
    """

    def __init__(self, nodes, weights, center, gauss_weights, gauss_center, order):
        self.nodes = nodes
        self.weights = weights
        self.center = center
        self.gauss_weights = gauss_weights
        self.gauss_center = gauss_center
        self.order = order

    def apply(self, f, lo, hi):
        """Return ``(integral, error, integral of |f|)`` over ``[lo, hi]``."""
        mid = 0.5 * (lo + hi)
        half = 0.5 * (hi - lo)
        f_mid = f(mid)
        kronrod = self.center * f_mid
        magnitude = self.center * abs(f_mid)
        gauss = self.gauss_center * f_mid
        pairs = [None] * len(self.nodes)
        # The sums accumulate in evaluation order, as QUADPACK's do.
        for index in self.order:
            offset = half * self.nodes[index]
            left, right = f(mid - offset), f(mid + offset)
            pairs[index] = (left, right)
            kronrod += self.weights[index] * (left + right)
            magnitude += self.weights[index] * (abs(left) + abs(right))
            if index % 2 == 1:
                gauss += self.gauss_weights[index // 2] * (left + right)
        mean = 0.5 * kronrod
        spread = self.center * abs(f_mid - mean)
        for weight, (left, right) in zip(self.weights, pairs):
            spread += weight * (abs(left - mean) + abs(right - mean))
        width = abs(half)
        error = abs((kronrod - gauss) * half)
        spread *= width
        magnitude *= width
        # QUADPACK's error estimate: the Gauss-Kronrod difference scaled by how smooth the
        # integrand looks, and never below what rounding in the sums can resolve.
        if spread != 0 and error != 0:
            error = spread * min(1.0, (200.0 * error / spread) ** 1.5)
        if magnitude > _TINY / (50.0 * _EPSILON):
            error = max(50.0 * _EPSILON * magnitude, error)
        return kronrod * half, error, magnitude


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
    order=(1, 3, 5, 7, 9, 0, 2, 4, 6, 8),
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
    order=(0, 1, 2, 3, 4, 5, 6),
)

_MESSAGES = {
    2: "The occurrence of roundoff error is detected, which prevents \n  the requested "
    "tolerance from being achieved.  The error may be \n  underestimated.",
    3: "Extremely bad integrand behavior occurs at some points of the\n  integration interval.",
    5: "The integral is probably divergent, or slowly convergent.",
}


def _limit_message(limit):
    return (
        f"The maximum number of subdivisions ({limit}) has been achieved.\n  If increasing the "
        "limit yields no improvement it is advised to analyze \n  the integrand in order to "
        "determine the difficulties.  If the position of a \n  local difficulty can be "
        "determined (singularity, discontinuity) one will \n  probably gain from splitting up "
        "the interval and calling the integrator \n  on the subranges.  Perhaps a "
        "special-purpose integrator should be used."
    )


def _real(value):
    """Convert an integrand value to ``float`` the way SciPy's compiled wrapper does."""
    if isinstance(value, (str, bytes, list, tuple, dict, complex)) or value is None:
        raise TypeError(f"must be real number, not {type(value).__name__}")
    return float(value)


def _wynn(sequence):
    """The last even-column entry of Wynn's epsilon table for ``sequence``.

    The even columns hold Shanks-transform estimates of the sequence's limit. Construction stops
    early when two neighbouring entries coincide, since the next column would divide by zero.
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
    """One piece of the range: its bounds, integral and error estimates, and bisection depth."""

    def __init__(self, lo, hi, area, error, level):
        self.lo = lo
        self.hi = hi
        self.area = area
        self.error = error
        self.level = level


class _Partition:
    """The pieces of the integration range, in QUADPACK's slot order.

    Bisecting a piece keeps the half with the larger error in its slot and appends the other,
    so ``full_output``'s ``alist`` and ``blist`` list pieces in the same order as SciPy's.
    ``magnitude`` estimates the integral of ``|f|`` over the whole range.
    """

    def __init__(self, f, rule, bounds, limit):
        self.f = f
        self.rule = rule
        self.limit = limit
        self.pieces = []
        self.magnitude = 0.0
        self.evaluations = 0
        self.roundoff = 0
        for lo, hi in zip(bounds, bounds[1:]):
            self.pieces.append(self._piece(lo, hi, 0))

    def _piece(self, lo, hi, level):
        area, error, magnitude = self.rule.apply(self.f, lo, hi)
        self.evaluations += 1
        if level == 0:
            self.magnitude += magnitude
        return _Piece(lo, hi, area, error, level)

    def area(self):
        return sum(piece.area for piece in self.pieces)

    def error(self):
        return sum(piece.error for piece in self.pieces)

    def worst(self, below_level=None):
        """The slot with the largest error, among pieces above ``below_level`` if given."""
        candidates = [
            index
            for index, piece in enumerate(self.pieces)
            if below_level is None or piece.level < below_level
        ]
        return max(candidates, key=lambda index: self.pieces[index].error, default=None)

    def bisect(self, index):
        """Split a piece in two; return a nonzero code when that is impossible or futile."""
        if len(self.pieces) >= self.limit:
            return 1
        piece = self.pieces[index]
        mid = 0.5 * (piece.lo + piece.hi)
        if not min(piece.lo, piece.hi) < mid < max(piece.lo, piece.hi):
            return 3
        left = self._piece(piece.lo, mid, piece.level + 1)
        right = self._piece(mid, piece.hi, piece.level + 1)
        keep, other = (right, left) if right.error > left.error else (left, right)
        self.pieces[index] = keep
        self.pieces.append(other)
        # A split that leaves the area unchanged without shrinking the error is rounding noise.
        area = left.area + right.area
        error = left.error + right.error
        if abs(piece.area - area) <= 1e-5 * abs(area) and error >= 0.99 * piece.error:
            self.roundoff += 1
            if self.roundoff >= 10:
                return 2
        return 0


def _adapt(partition, epsabs, epsrel):
    """Refine ``partition`` until it meets the tolerance; return ``(result, error, code)``.

    The codes are SciPy's ``ier``: 0 success, 1 subdivision limit, 2 roundoff, 3 a piece too
    small to bisect, and 5 probable divergence.
    """

    def tolerance(value):
        return max(epsabs, epsrel * abs(value))

    area, error = partition.area(), partition.error()
    if not math.isfinite(area + error):
        return area, error, 2
    if error <= tolerance(area):
        return area, error, 0
    # Pieces bisected at least `small_level` times count as small. Once refinement reaches
    # them, the error sits near a point, and extrapolating successive totals takes over.
    small_level = 2
    totals, estimates, best = [area], [], None
    code = 0
    while code == 0:
        index = partition.worst()
        split_level = partition.pieces[index].level
        code = partition.bisect(index)
        area, error = partition.area(), partition.error()
        if code or error <= tolerance(area) or not math.isfinite(area + error):
            break
        if split_level < small_level:
            continue
        # Settle the large pieces first, so the totals differ only near the difficulty.
        while code == 0:
            large = partition.worst(below_level=small_level)
            large_error = sum(
                piece.error for piece in partition.pieces if piece.level < small_level
            )
            if large is None or large_error <= tolerance(partition.area()):
                break
            code = partition.bisect(large)
        area, error = partition.area(), partition.error()
        totals.append(area)
        if len(totals) >= 3:
            estimates.append(_wynn(totals[-_WYNN_TERMS:]))
            if len(estimates) >= 3:
                estimate = estimates[-1]
                spread = abs(estimate - estimates[-2]) + abs(estimate - estimates[-3])
                spread = max(spread, 5.0 * _EPSILON * abs(estimate))
                if best is None or spread < best[1]:
                    best = (estimate, spread)
                if spread <= tolerance(estimate):
                    return _extrapolated(partition, estimate, spread, 0)
        small_level += 1
    if not math.isfinite(area + error):
        return area, error, 2
    if code == 0 or best is None or best[1] >= error:
        return area, error, code
    return _extrapolated(partition, *best, code)


def _extrapolated(partition, estimate, spread, code):
    """Return an extrapolated result, flagged as divergent when it is far from the totals."""
    area = partition.area()
    ratio = abs(estimate) / max(abs(area), _TINY)
    disagree = (estimate > 0) != (area > 0) or not 0.01 <= ratio <= 100.0
    if disagree and max(abs(estimate), abs(area)) > 0.01 * partition.magnitude:
        return estimate, spread, 5
    return estimate, spread, code


def _infinite_integrand(f, a, b):
    """Map an infinite range onto ``(0, 1]`` with ``x = (1 - t) / t``."""
    if a == -math.inf and b == math.inf:
        return lambda t: (f((1.0 - t) / t) + f(-(1.0 - t) / t)) / t / t, 2
    if b == math.inf:
        return lambda t: f(a + (1.0 - t) / t) / t / t, 1
    return lambda t: f(b - (1.0 - t) / t) / t / t, 1


def _full_output(partition, evaluations, limit, breakpoints):
    """SciPy's ``infodict``; array entries past ``last`` are zero."""
    pieces = partition.pieces if partition is not None else []
    count = len(pieces)

    def padded(values, dtype):
        array = np.zeros(limit, dtype=dtype)
        array[:count] = values
        return array

    by_error = sorted(range(count), key=lambda index: (-pieces[index].error, -index))
    info = {
        "neval": evaluations,
        "last": count,
        "iord": padded(by_error, np.int32),
        "alist": padded([piece.lo for piece in pieces], np.float64),
        "blist": padded([piece.hi for piece in pieces], np.float64),
        "rlist": padded([piece.area for piece in pieces], np.float64),
        "elist": padded([piece.error for piece in pieces], np.float64),
    }
    if breakpoints is not None:
        info["pts"] = np.array(breakpoints, dtype=np.float64)
        info["level"] = padded([piece.level for piece in pieces], np.int32)
        info["ndin"] = np.zeros(len(breakpoints), dtype=np.int32)
    return info


def quad(
    func,
    a,
    b,
    args=(),
    full_output=0,
    epsabs=1.49e-8,
    epsrel=1.49e-8,
    limit=50,
    points=None,
    weight=None,
    wvar=None,
    wopts=None,
    maxp1=50,
    limlst=50,
    complex_func=False,
):
    """Integrate ``func(x, *args)`` from ``a`` to ``b``; either bound may be infinite.

    Returns ``(result, abserr)``, followed by an ``infodict`` of diagnostics when
    ``full_output`` is true. When the tolerance cannot be met, an ``IntegrationWarning``
    explains why, or with ``full_output`` the explanation is returned as a fourth element.
    ``points`` lists places inside a finite range where the integrand misbehaves.

    >>> quad(lambda x: x * x, 0, 1)
    (0.33333333333333337, 3.700743415417189e-15)
    """
    if not isinstance(args, tuple):
        args = (args,)
    if weight is not None:
        raise NotImplementedError(
            "scipy.integrate.quad(weight=...) is not supported by shellsim's SciPy"
        )
    if complex_func:
        real = quad(
            lambda x, *rest: np.real(func(x, *rest)),
            a, b, args, full_output, epsabs, epsrel, limit, points,
        )
        imag = quad(
            lambda x, *rest: np.imag(func(x, *rest)),
            a, b, args, full_output, epsabs, epsrel, limit, points,
        )
        result = (real[0] + 1j * imag[0], real[1] + 1j * imag[1])
        if full_output:
            return result + ({"real": real[2:], "imag": imag[2:]},)
        return result
    if not callable(func):
        raise ValueError("invalid callable given")
    flip = b < a
    if flip:
        a, b = b, a
    a, b = float(a), float(b)
    if limit < 1:
        raise ValueError("Invalid 'limit' argument. There must be at least one subinterval")
    if epsabs <= 0 and epsrel < max(50 * _EPSILON, 5e-29):
        raise ValueError(
            "If 'epsabs'<=0, 'epsrel' must be greater than both 5e-29 and 50*(machine epsilon)."
        )
    infinite = math.isinf(a) or math.isinf(b)
    if infinite and points is not None:
        raise ValueError("Infinity inputs cannot be used with break points.")

    def f(x):
        return _real(func(x, *args))

    breakpoints = None
    partition = None
    if a == b:
        result, error, code, evaluations = 0.0, 0.0, 0, 0
    else:
        if infinite:
            integrand, calls = _infinite_integrand(f, a, b)
            partition = _Partition(integrand, _KRONROD15, (0.0, 1.0), limit)
            calls *= 15
        else:
            bounds = [a, b]
            if points is not None:
                inside = sorted({float(point) for point in points if a < point < b})
                bounds = breakpoints = [a, *inside, b]
            partition = _Partition(f, _KRONROD21, bounds, limit)
            calls = 21
        result, error, code = _adapt(partition, epsabs, epsrel)
        evaluations = calls * partition.evaluations
    if flip:
        result = -result
    output = (result, error)
    if full_output:
        output += (_full_output(partition, evaluations, limit, breakpoints),)
    if code == 0:
        return output
    message = _limit_message(limit) if code == 1 else _MESSAGES[code]
    if full_output:
        return output + (message,)
    warnings.warn(message, IntegrationWarning, stacklevel=2)
    return output


def nquad(func, ranges, args=None, opts=None, full_output=False):
    """Integrate ``func(x0, x1, ..., *args)`` over nested ranges, innermost ``x0`` first.

    Each entry of ``ranges`` is a ``(lo, hi)`` pair or a callable of the outer variables and
    ``args`` returning one; ``opts`` gives ``quad`` options the same way, one per level or one
    for all. With ``full_output``, a third element counts integrand evaluations.

    >>> nquad(lambda x, y: x * y, [[0, 1], [0, 2]])[0]
    0.9999999999999999
    """
    depth = len(ranges)
    args = () if args is None else tuple(args)
    if opts is None or isinstance(opts, dict) or callable(opts):
        opts = [opts or {}] * depth
    calls = [0]

    def integrate(level, outer):
        bounds = ranges[level]
        if callable(bounds):
            bounds = bounds(*outer, *args)
        options = opts[level]
        if callable(options):
            options = options(*outer, *args)
        if level == 0:

            def integrand(x):
                calls[0] += 1
                return func(x, *outer, *args)

        else:

            def integrand(x):
                return integrate(level - 1, (x, *outer))[0]

        return quad(integrand, bounds[0], bounds[1], **options)

    result = integrate(depth - 1, ())
    if full_output:
        return result[0], result[1], {"neval": calls[0]}
    return result[0], result[1]


def dblquad(func, a, b, gfun, hfun, args=(), epsabs=1.49e-8, epsrel=1.49e-8):
    """Integrate ``func(y, x, *args)`` for ``x`` in ``[a, b]`` and ``y`` from ``gfun(x)`` to
    ``hfun(x)``; the inner bounds may also be constants.

    >>> dblquad(lambda y, x: x * y, 0, 1, 0, lambda x: x)[0]
    0.125
    """

    def inner(x, *rest):
        return (gfun(x) if callable(gfun) else gfun, hfun(x) if callable(hfun) else hfun)

    options = {"epsabs": epsabs, "epsrel": epsrel}
    return nquad(func, [inner, [a, b]], args=args, opts=options)


def tplquad(func, a, b, gfun, hfun, qfun, rfun, args=(), epsabs=1.49e-8, epsrel=1.49e-8):
    """Integrate ``func(z, y, x, *args)`` for ``x`` in ``[a, b]``, ``y`` from ``gfun(x)`` to
    ``hfun(x)``, and ``z`` from ``qfun(x, y)`` to ``rfun(x, y)``.

    >>> tplquad(lambda z, y, x: 1.0, 0, 1, 0, 1, 0, 1)[0]
    1.0
    """

    def middle(x, *rest):
        return (gfun(x) if callable(gfun) else gfun, hfun(x) if callable(hfun) else hfun)

    def inner(y, x, *rest):
        return (qfun(x, y) if callable(qfun) else qfun, rfun(x, y) if callable(rfun) else rfun)

    options = {"epsabs": epsabs, "epsrel": epsrel}
    return nquad(func, [inner, middle, [a, b]], args=args, opts=options)
