"""The ``rv_continuous``/``rv_discrete`` distribution framework and its built-in distributions.

A distribution class defines its shape in "standard form" (``loc=0``, ``scale=1`` for continuous
distributions; ``loc=0`` for discrete ones) through a handful of private hooks: ``_pdf``/``_pmf``,
``_cdf``, ``_ppf``, ``_stats`` (raw mean/variance/skewness/excess kurtosis), ``_entropy`` and
``_rvs``. The base classes turn those into the public API (``pdf``, ``cdf``, ``sf``, ``ppf``,
``isf``, their logarithms, ``stats``, ``moment``, ``entropy``, ``interval``, ``support``, ``rvs``,
frozen distributions) and apply ``loc``/``scale``. A hook a subclass does not define falls back to
a generic implementation derived from whichever of ``_pdf``/``_cdf``/``_ppf`` *is* defined
(numerical differentiation, integration or bisection), mirroring SciPy's own fallbacks but with a
single simple implementation of each instead of SciPy's layered dispatch.

A subclass's shape parameters are read from the declared parameters of its ``_pdf``/``_pmf``
and ``_cdf`` methods with ``inspect.signature``, as in SciPy. The public methods themselves are generated once per instance with ``exec`` so
that shellsim's own CPython-compatible argument binding produces SciPy's argument-count and
keyword-argument error messages, instead of this module reproducing that text by hand.
"""

import inspect
import math

import numpy as np

from scipy import special
from scipy.special._ufuncs import _binom_cdf, _binom_isf, _binom_pmf, _binom_ppf
from scipy._lib._util import check_random_state

__all__ = [
    "rv_continuous",
    "rv_discrete",
    "norm",
    "t",
    "chi2",
    "f",
    "uniform",
    "expon",
    "binom",
    "poisson",
]


def _shape_tuple_literal(names):
    if not names:
        return "()"
    if len(names) == 1:
        return f"({names[0]},)"
    return "(" + ", ".join(names) + ")"


def _make_binder(shape_names, leading, trailing=(), scale=True):
    """Build a real function with the given parameter list via ``exec``.

    ``leading`` are required positional names before the shape parameters (the domain variable
    for ``pdf``/``cdf``/... , or the moment order for ``moment``); ``trailing`` are
    ``(name, default)`` pairs for keyword-only parameters after ``loc``/``scale`` (``stats``'
    ``moments=``). When ``scale`` is false (discrete distributions), the generated function still
    returns a `scale` slot so callers can share code with the continuous case, but it is always
    the literal ``1`` rather than a real parameter.
    """
    params = ["self", *leading, *shape_names, "loc=0"]
    if scale:
        params.append("scale=1")
    if trailing:
        params.append("*")
        params.extend(f"{name}={default!r}" for name, default in trailing)
    ret = [*leading, _shape_tuple_literal(shape_names), "loc", "scale" if scale else "1"]
    ret.extend(name for name, _ in trailing)
    source = f"def _bind({', '.join(params)}):\n    return ({', '.join(ret)})\n"
    exec(source)
    return _bind


def _comb(n, k):
    """Small integer binomial coefficient, used to expand raw moments; avoids depending on
    whether shellsim's ``math`` module has ``math.comb``."""
    result = 1
    for i in range(k):
        result = result * (n - i) // (i + 1)
    return result


class rv_generic:
    """Shared machinery for ``rv_continuous`` and ``rv_discrete``.

    Subclasses set ``_shape_methods`` (the private method names shape inference reads) and
    ``_has_scale``, and implement ``_generic_method`` to return their own base class's version of
    a hook (so shape inference can tell an override from the generic fallback), so the same code
    here can serve both.
    """

    _has_scale = True
    _shape_methods = ()

    def _generic_method(self, name):
        raise NotImplementedError

    def __init__(self, a=None, b=None, name=None, shapes=None, badvalue=np.nan):
        self.a = (-np.inf if self._has_scale else 0.0) if a is None else a
        self.b = np.inf if b is None else b
        self.name = name
        self.badvalue = badvalue
        self.shapes, shape_names = self._infer_shapes(shapes)
        self.numargs = len(shape_names)
        self._shape_names = shape_names
        self._value_binder = _make_binder(shape_names, ["v"], scale=self._has_scale)
        self._shape_binder = _make_binder(shape_names, [], scale=self._has_scale)
        self._stats_binder = _make_binder(
            shape_names, [], trailing=[("moments", "mv")], scale=self._has_scale
        )

    def _infer_shapes(self, explicit_shapes):
        if explicit_shapes is not None:
            text = explicit_shapes.strip()
            names = [part.strip() for part in text.split(",")] if text else []
            return (text if names else None), names

        candidates = []
        cls = type(self)
        for method_name in self._shape_methods:
            if getattr(cls, method_name, None) is self._generic_method(method_name):
                continue
            params = list(inspect.signature(getattr(self, method_name)).parameters.values())
            if not params:
                continue
            names = []
            for param in params[1:]:
                if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
                    raise TypeError("*args are not allowed w/out explicit shapes")
                if param.default is not inspect.Parameter.empty:
                    raise TypeError(f"defaults are not allowed for shapes: {param.name}=...")
                names.append(param.name)
            candidates.append(names)

        nonempty = [names for names in candidates if names]
        if not nonempty:
            return None, []
        first = nonempty[0]
        for other in nonempty[1:]:
            if other != first:
                raise TypeError("Shape arguments are inconsistent.")
        return ", ".join(first), first

    def __call__(self, *args, **kwds):
        return _FrozenDist(self, args, kwds)

    def __repr__(self):
        return f"<scipy.stats._distributions.{type(self).__name__} object at {id(self):#x}>"

    # -- Shared numeric helpers ------------------------------------------------------------
    def _out(self, result):
        result = np.asarray(result)
        return result[()] if result.ndim == 0 else result

    def _argcheck(self, *shapes):
        ok = True
        for shape in shapes:
            ok = ok & (np.asarray(shape, dtype=float) > 0)
        return ok

    def _valid(self, shapes, scale):
        valid = np.asarray(scale, dtype=float) > 0
        if shapes:
            valid = valid & np.asarray(self._argcheck(*shapes))
        return valid

    def _get_support(self, *shapes):
        return self.a, self.b

    # -- Methods shared between continuous and discrete distributions ----------------------
    def stats(self, *args, **kwds):
        shapes, loc, scale, moments = self._stats_binder(self, *args, **kwds)
        loc = np.asarray(loc, dtype=float)
        scale = np.asarray(scale, dtype=float)
        valid = self._valid(shapes, scale)
        mean0, var0, skew0, kurt0 = (np.asarray(v, dtype=float) for v in self._stats(*shapes))
        results = []
        for letter in "mvsk":
            if letter not in moments:
                continue
            if letter == "m":
                value = loc + scale * mean0
            elif letter == "v":
                value = scale**2 * var0
            elif letter == "s":
                value = skew0
            else:
                value = kurt0
            results.append(self._out(np.where(valid, value, self.badvalue)))
        return tuple(results)

    def mean(self, *args, **kwds):
        return self.stats(*args, **kwds, moments="m")[0]

    def var(self, *args, **kwds):
        return self.stats(*args, **kwds, moments="v")[0]

    def std(self, *args, **kwds):
        return np.sqrt(self.stats(*args, **kwds, moments="v")[0])

    def median(self, *args, **kwds):
        return self.ppf(0.5, *args, **kwds)

    def support(self, *args, **kwds):
        shapes, loc, scale = self._shape_binder(self, *args, **kwds)
        a, b = self._get_support(*shapes)
        loc = np.asarray(loc, dtype=float)
        scale = np.asarray(scale, dtype=float)
        return self._out(loc + scale * a), self._out(loc + scale * b)

    def interval(self, confidence, *args, **kwds):
        alpha = (1.0 - np.asarray(confidence, dtype=float)) / 2.0
        return self.ppf(alpha, *args, **kwds), self.ppf(1.0 - alpha, *args, **kwds)

    def entropy(self, *args, **kwds):
        shapes, loc, scale = self._shape_binder(self, *args, **kwds)
        scale = np.asarray(scale, dtype=float)
        valid = self._valid(shapes, scale)
        with np.errstate(divide="ignore"):
            h = np.asarray(self._entropy(*shapes), dtype=float) + np.log(np.abs(scale))
        return self._out(np.where(valid, h, self.badvalue))

    def moment(self, order, *args, **kwds):
        order0, shapes, loc, scale = self._value_binder(self, order, *args, **kwds)
        n = order0
        if not float(n).is_integer() or not (1 <= n <= 4):
            raise NotImplementedError(f"moments of order {n} are not supported by shellsim's SciPy")
        n = int(n)
        mean0, var0, skew0, kurt0 = (np.asarray(v, dtype=float) for v in self._stats(*shapes))
        central = [np.ones_like(mean0), np.zeros_like(mean0), var0, skew0 * var0**1.5, (kurt0 + 3.0) * var0**2]
        raw = [np.ones_like(mean0), mean0]
        raw.append(central[2] + mean0**2)
        raw.append(central[3] + 3 * mean0 * central[2] + mean0**3)
        raw.append(central[4] + 4 * mean0 * central[3] + 6 * mean0**2 * central[2] + mean0**4)
        loc = np.asarray(loc, dtype=float)
        scale = np.asarray(scale, dtype=float)
        total = sum(_comb(n, k) * loc ** (n - k) * scale**k * raw[k] for k in range(n + 1))
        valid = self._valid(shapes, scale)
        return self._out(np.where(valid, total, self.badvalue))


class _FrozenDist:
    """A distribution with its shape/``loc``/``scale`` arguments fixed, as ``dist(...)`` returns.

    Every call forwards to the underlying distribution with the frozen positional arguments
    appended after whatever the caller passes (the value for ``pdf``/``cdf``/..., or nothing for
    ``mean``/``std``/...) and the frozen keywords merged under the caller's. shellsim's object
    model does not call ``__getattr__`` for missing instance attributes, so each forwarding
    method is built as a real instance attribute up front instead of looked up lazily.
    """

    _METHODS = (
        "pdf", "logpdf", "cdf", "logcdf", "sf", "logsf", "ppf", "isf",
        "pmf", "logpmf",
        "stats", "entropy", "mean", "var", "std", "median", "support", "interval", "moment",
        "rvs", "nnlf", "fit", "expect",
    )

    def __init__(self, dist, args, kwds):
        self._dist = dist
        self._args = args
        self._kwds = kwds
        for name in self._METHODS:
            method = getattr(dist, name, None)
            if method is not None:
                setattr(self, name, self._forward(method))

    def _forward(self, method):
        def call(*args, **kwds):
            merged = dict(self._kwds)
            merged.update(kwds)
            return method(*args, *self._args, **merged)

        return call


# ----------------------------------------------------------------------------------------------
# Continuous distributions
# ----------------------------------------------------------------------------------------------


class rv_continuous(rv_generic):
    """Base class for continuous probability distributions.

    Subclasses normally override ``_pdf`` and/or ``_cdf`` (and ``_ppf``, ``_stats``, ``_entropy``
    and ``_rvs`` for exactness and performance); this class derives whichever of them is missing
    from the ones that are present.
    """

    _has_scale = True
    _shape_methods = ("_pdf", "_cdf")

    def _generic_method(self, name):
        return getattr(rv_continuous, name)

    # -- Generic fallbacks, used when a subclass does not override the hook ----------------
    def _pdf(self, x, *shapes):
        dx = 1e-5
        return (self._cdf(x + dx, *shapes) - self._cdf(x - dx, *shapes)) / (2.0 * dx)

    def _cdf(self, x, *shapes):
        def pdf_at(t):
            return float(self._pdf(np.asarray(t), *shapes))

        a = self.a if np.isfinite(self.a) else -50.0
        results = [_simpson(pdf_at, a, float(xi)) for xi in np.atleast_1d(x)]
        result = np.array(results).reshape(np.shape(x))
        return result[()] if result.ndim == 0 else result

    def _ppf(self, q, *shapes):
        def cdf_at(t):
            return float(self._cdf(np.asarray(t), *shapes))

        results = [_bisect_ppf(cdf_at, float(qi), self.a, self.b) for qi in np.atleast_1d(q)]
        result = np.array(results).reshape(np.shape(q))
        return result[()] if result.ndim == 0 else result

    def _stats(self, *shapes):
        return np.nan, np.nan, np.nan, np.nan

    def _entropy(self, *shapes):
        def integrand(t):
            p = float(self._pdf(np.asarray(t), *shapes))
            return -p * math.log(p) if p > 0 else 0.0

        a = self.a if np.isfinite(self.a) else -50.0
        b = self.b if np.isfinite(self.b) else 50.0
        return _simpson(integrand, a, b)

    def _rvs(self, rng, size, *shapes):
        u = rng.random_sample(size=size)
        return self._ppf(np.asarray(u, dtype=float), *shapes)

    # -- Public API --------------------------------------------------------------------------
    def pdf(self, x, *args, **kwds):
        x0, shapes, loc, scale = self._value_binder(self, x, *args, **kwds)
        x0 = np.asarray(x0, dtype=float)
        loc = np.asarray(loc, dtype=float)
        scale = np.asarray(scale, dtype=float)
        safe_scale = np.where(scale > 0, scale, 1.0)
        xs = (x0 - loc) / safe_scale
        a, b = self._get_support(*shapes)
        valid = self._valid(shapes, scale)
        with np.errstate(all="ignore"):
            inside = (xs >= a) & (xs <= b)
            density = np.where(inside, self._pdf(xs, *shapes) / safe_scale, 0.0)
            result = np.where(valid, density, self.badvalue)
        return self._out(result)

    def logpdf(self, x, *args, **kwds):
        with np.errstate(divide="ignore"):
            return np.log(self.pdf(x, *args, **kwds))

    def cdf(self, x, *args, **kwds):
        x0, shapes, loc, scale = self._value_binder(self, x, *args, **kwds)
        x0 = np.asarray(x0, dtype=float)
        loc = np.asarray(loc, dtype=float)
        scale = np.asarray(scale, dtype=float)
        safe_scale = np.where(scale > 0, scale, 1.0)
        xs = (x0 - loc) / safe_scale
        a, b = self._get_support(*shapes)
        valid = self._valid(shapes, scale)
        with np.errstate(all="ignore"):
            below = xs < a
            above = xs > b
            result = np.where(above, 1.0, np.where(below, 0.0, self._cdf(xs, *shapes)))
            result = np.where(valid, result, self.badvalue)
        return self._out(result)

    def logcdf(self, x, *args, **kwds):
        with np.errstate(divide="ignore"):
            return np.log(self.cdf(x, *args, **kwds))

    def sf(self, x, *args, **kwds):
        return 1.0 - np.asarray(self.cdf(x, *args, **kwds), dtype=float)

    def logsf(self, x, *args, **kwds):
        with np.errstate(divide="ignore"):
            return np.log(self.sf(x, *args, **kwds))

    def ppf(self, q, *args, **kwds):
        q0, shapes, loc, scale = self._value_binder(self, q, *args, **kwds)
        q0 = np.asarray(q0, dtype=float)
        loc = np.asarray(loc, dtype=float)
        scale = np.asarray(scale, dtype=float)
        a, b = self._get_support(*shapes)
        valid = self._valid(shapes, scale) & (q0 >= 0.0) & (q0 <= 1.0)
        with np.errstate(all="ignore"):
            clipped = np.clip(q0, 0.0, 1.0)
            core = np.where(q0 <= 0.0, a, np.where(q0 >= 1.0, b, self._ppf(clipped, *shapes)))
            result = np.where(valid, loc + scale * core, self.badvalue)
        return self._out(result)

    def isf(self, q, *args, **kwds):
        return self.ppf(1.0 - np.asarray(q, dtype=float), *args, **kwds)

    def rvs(self, *args, size=None, random_state=None, **kwds):
        loc = kwds.pop("loc", 0)
        scale = kwds.pop("scale", 1)
        if kwds or len(args) != self.numargs:
            raise TypeError(f"{self.name}.rvs() got an unexpected argument")
        rng = check_random_state(random_state)
        raw = self._rvs(rng, size, *args)
        return loc + scale * np.asarray(raw)

    def nnlf(self, theta, x):
        theta = tuple(theta)
        shapes, loc, scale = theta[:-2], theta[-2], theta[-1]
        if scale <= 0:
            return np.inf
        x = np.asarray(x, dtype=float)
        xs = (x - loc) / scale
        a, b = self._get_support(*shapes)
        if np.any((xs < a) | (xs > b)):
            return np.inf
        with np.errstate(all="ignore"):
            logpdf = np.log(self._pdf(xs, *shapes)) - math.log(scale)
        return -np.sum(logpdf)

    def fit(self, data, *args, **kwds):
        # Generic fitting maximizes the likelihood numerically; only the closed-form overrides
        # below (norm, uniform, expon) avoid needing `scipy.optimize`.
        import scipy.optimize  # noqa: F401

    def expect(self, func=None, args=(), loc=0, scale=1, lb=None, ub=None, conditional=False, **kwds):
        # SciPy computes this by numerical integration; shellsim does not provide scipy.integrate.
        import scipy.integrate  # noqa: F401


def _simpson(func, lo, hi, n=200):
    """A fixed-resolution composite Simpson's rule, the generic fallback's numerical integrator.

    Used only when a distribution does not override ``_cdf``/``_entropy``; every built-in
    distribution below has a closed form, so this never runs during the portable test suite.
    """
    if hi <= lo:
        return 0.0
    xs = np.linspace(lo, hi, 2 * n + 1)
    ys = np.array([func(x) for x in xs])
    h = (hi - lo) / (2 * n)
    return h / 3.0 * (ys[0] + ys[-1] + 4.0 * np.sum(ys[1:-1:2]) + 2.0 * np.sum(ys[2:-2:2]))


def _bisect_ppf(cdf_at, q, a, b):
    """Bisect for the generic ``_ppf`` fallback, expanding an infinite bound until it brackets."""
    lo = a if np.isfinite(a) else -1.0
    hi = b if np.isfinite(b) else 1.0
    while not np.isfinite(a) and cdf_at(lo) > q:
        lo *= 2.0
    while not np.isfinite(b) and cdf_at(hi) < q:
        hi *= 2.0
    for _ in range(60):
        mid = (lo + hi) / 2.0
        if cdf_at(mid) < q:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


class _norm_gen(rv_continuous):
    def _pdf(self, x):
        return np.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)

    def _cdf(self, x):
        return special.ndtr(x)

    def _ppf(self, q):
        return special.ndtri(q)

    def _stats(self):
        return 0.0, 1.0, 0.0, 0.0

    def _entropy(self):
        return 0.5 * math.log(2.0 * math.pi * math.e)

    def _rvs(self, rng, size):
        return rng.standard_normal(size=size)


class _t_gen(rv_continuous):
    def _pdf(self, x, df):
        return np.exp(
            -0.5 * (df + 1.0) * np.log1p(x * x / df) - 0.5 * np.log(df) - special.betaln(0.5, 0.5 * df)
        )

    def _cdf(self, x, df):
        return special.stdtr(df, x)

    def _ppf(self, q, df):
        return special.stdtrit(df, q)

    def _stats(self, df):
        mean = np.where(df > 1, 0.0, np.nan)
        var = np.where(df > 2, df / (df - 2.0), np.where(df > 1, np.inf, np.nan))
        skew = np.where(df > 3, 0.0, np.nan)
        kurt = np.where(df > 4, 6.0 / (df - 4.0), np.where(df > 2, np.inf, np.nan))
        return mean, var, skew, kurt

    def _entropy(self, df):
        half = 0.5 * df
        return (
            (df + 1.0) / 2.0 * (special.psi((df + 1.0) / 2.0) - special.psi(half))
            + 0.5 * math.log(df)
            + special.betaln(half, 0.5)
        )

    def _rvs(self, rng, size, df):
        return rng.standard_t(df, size=size)


class _chi2_gen(rv_continuous):
    def _pdf(self, x, df):
        with np.errstate(divide="ignore", invalid="ignore"):
            logpdf = (0.5 * df - 1.0) * np.log(x) - 0.5 * x - 0.5 * df * math.log(2.0) - special.gammaln(0.5 * df)
        return np.exp(logpdf)

    def _cdf(self, x, df):
        return special.chdtr(df, x)

    def _ppf(self, q, df):
        return special.chdtri(df, 1.0 - q)

    def _stats(self, df):
        return df, 2.0 * df, np.sqrt(8.0 / df), 12.0 / df

    def _entropy(self, df):
        half = 0.5 * df
        return half + math.log(2.0) + special.gammaln(half) + (1.0 - half) * special.psi(half)

    def _rvs(self, rng, size, df):
        return rng.chisquare(df, size=size)


class _f_gen(rv_continuous):
    def _pdf(self, x, dfn, dfd):
        with np.errstate(divide="ignore", invalid="ignore"):
            logpdf = (
                0.5 * dfn * math.log(dfn)
                + 0.5 * dfd * math.log(dfd)
                + (0.5 * dfn - 1.0) * np.log(x)
                - 0.5 * (dfn + dfd) * np.log(dfd + dfn * x)
                - special.betaln(0.5 * dfn, 0.5 * dfd)
            )
        return np.exp(logpdf)

    def _cdf(self, x, dfn, dfd):
        return special.fdtr(dfn, dfd, x)

    def _ppf(self, q, dfn, dfd):
        # Unlike `chdtri`, `fdtri` inverts `fdtr` (the lower CDF) directly, not `fdtrc`.
        return special.fdtri(dfn, dfd, q)

    def _stats(self, dfn, dfd):
        mean = np.where(dfd > 2, dfd / (dfd - 2.0), np.nan)
        var = np.where(
            dfd > 4,
            2.0 * dfd**2 * (dfn + dfd - 2.0) / (dfn * (dfd - 2.0) ** 2 * (dfd - 4.0)),
            np.nan,
        )
        skew = np.where(
            dfd > 6,
            (2.0 * dfn + dfd - 2.0) * np.sqrt(8.0 * (dfd - 4.0)) / ((dfd - 6.0) * np.sqrt(dfn * (dfn + dfd - 2.0))),
            np.nan,
        )
        kurt = np.where(
            dfd > 8,
            12.0
            * (dfn * (5.0 * dfd - 22.0) * (dfn + dfd - 2.0) + (dfd - 4.0) * (dfd - 2.0) ** 2)
            / (dfn * (dfd - 6.0) * (dfd - 8.0) * (dfn + dfd - 2.0)),
            np.nan,
        )
        return mean, var, skew, kurt

    def _entropy(self, dfn, dfd):
        return (
            math.log(dfd / dfn)
            + special.betaln(0.5 * dfn, 0.5 * dfd)
            + (1.0 - 0.5 * dfn) * special.psi(0.5 * dfn)
            - (1.0 + 0.5 * dfd) * special.psi(0.5 * dfd)
            + 0.5 * (dfn + dfd) * special.psi(0.5 * (dfn + dfd))
        )

    def _rvs(self, rng, size, dfn, dfd):
        return rng.f(dfn, dfd, size=size)


class _uniform_gen(rv_continuous):
    def _pdf(self, x):
        return np.ones_like(x)

    def _cdf(self, x):
        return x

    def _ppf(self, q):
        return q

    def _stats(self):
        return 0.5, 1.0 / 12.0, 0.0, -1.2

    def _entropy(self):
        return 0.0

    def _rvs(self, rng, size):
        return rng.random_sample(size=size)

    def fit(self, data, *, floc=None, fscale=None):
        data = np.asarray(data, dtype=float)
        loc = float(np.min(data)) if floc is None else float(floc)
        scale = (float(np.max(data)) - loc) if fscale is None else float(fscale)
        return loc, scale


class _expon_gen(rv_continuous):
    def _pdf(self, x):
        return np.exp(-x)

    def _cdf(self, x):
        return -np.expm1(-x)

    def _ppf(self, q):
        return -np.log1p(-q)

    def _stats(self):
        return 1.0, 1.0, 2.0, 6.0

    def _entropy(self):
        return 1.0

    def _rvs(self, rng, size):
        return rng.standard_exponential(size=size)

    def fit(self, data, *, floc=None, fscale=None):
        data = np.asarray(data, dtype=float)
        loc = float(np.min(data)) if floc is None else float(floc)
        scale = (float(np.mean(data)) - loc) if fscale is None else float(fscale)
        return loc, scale


class _norm_gen_fit(_norm_gen):
    def fit(self, data, *, floc=None, fscale=None):
        data = np.asarray(data, dtype=float)
        loc = float(np.mean(data)) if floc is None else float(floc)
        if fscale is None:
            scale = float(np.sqrt(np.mean((data - loc) ** 2)))
        else:
            scale = float(fscale)
        return loc, scale



norm = _norm_gen_fit(name="norm")
t = _t_gen(name="t")
chi2 = _chi2_gen(a=0.0, name="chi2")
f = _f_gen(a=0.0, name="f")
uniform = _uniform_gen(a=0.0, b=1.0, name="uniform")
expon = _expon_gen(a=0.0, name="expon")


# ----------------------------------------------------------------------------------------------
# Discrete distributions
# ----------------------------------------------------------------------------------------------


class rv_discrete(rv_generic):
    """Base class for discrete probability distributions (``loc`` only, no ``scale``)."""

    _has_scale = False
    _shape_methods = ("_pmf", "_cdf")

    def _generic_method(self, name):
        return getattr(rv_discrete, name)

    def __init__(self, a=0, b=np.inf, name=None, shapes=None, badvalue=np.nan, values=None):
        if values is not None:
            raise NotImplementedError("rv_discrete(values=...) is not supported by shellsim's SciPy")
        super().__init__(a=a, b=b, name=name, shapes=shapes, badvalue=badvalue)

    def _pmf(self, k, *shapes):
        return self._cdf(k, *shapes) - self._cdf(k - 1.0, *shapes)

    def _cdf(self, k, *shapes):
        a, _ = self._get_support(*shapes)
        low = int(a) if np.isfinite(a) else 0

        def at(ki):
            ki = int(ki)
            return float(sum(float(self._pmf(np.asarray(float(j)), *shapes)) for j in range(low, ki + 1)))

        results = [at(ki) for ki in np.atleast_1d(k)]
        result = np.array(results).reshape(np.shape(k))
        return result[()] if result.ndim == 0 else result

    def _ppf(self, q, *shapes):
        a, b = self._get_support(*shapes)

        def search(qi):
            lo = int(a) if np.isfinite(a) else 0
            k = lo
            step = 1
            while float(self._cdf(np.asarray(float(k)), *shapes)) < qi:
                k += step
                step *= 2
                if np.isfinite(b) and k >= b:
                    return float(b)
            hi = k
            lo2 = max(lo, k - step)
            while lo2 < hi:
                mid = (lo2 + hi) // 2
                if float(self._cdf(np.asarray(float(mid)), *shapes)) >= qi:
                    hi = mid
                else:
                    lo2 = mid + 1
            return float(lo2)

        results = [search(float(qi)) for qi in np.atleast_1d(q)]
        result = np.array(results).reshape(np.shape(q))
        return result[()] if result.ndim == 0 else result

    def _stats(self, *shapes):
        return np.nan, np.nan, np.nan, np.nan

    def _entropy(self, *shapes):
        a, b = self._get_support(*shapes)
        low = int(a) if np.isfinite(a) else 0
        high = int(b) if np.isfinite(b) else low + 100000
        total = 0.0
        zero_run = 0
        k = low
        while k <= high:
            p = float(self._pmf(np.asarray(float(k)), *shapes))
            if p > 0.0:
                total -= p * math.log(p)
                zero_run = 0
            else:
                zero_run += 1
                if zero_run > 10 and k > low:
                    break
            k += 1
        return total

    def _rvs(self, rng, size, *shapes):
        u = rng.random_sample(size=size)
        return self._ppf(np.asarray(u, dtype=float), *shapes)

    def pmf(self, k, *args, **kwds):
        k0, shapes, loc, _scale = self._value_binder(self, k, *args, **kwds)
        k0 = np.asarray(k0, dtype=float)
        loc = np.asarray(loc, dtype=float)
        ks = np.floor(k0 - loc)
        a, b = self._get_support(*shapes)
        valid = self._valid(shapes, np.array(1.0))
        with np.errstate(all="ignore"):
            inside = (ks >= a) & (ks <= b) & (k0 - loc == ks)
            result = np.where(inside, self._pmf(ks, *shapes), 0.0)
            result = np.where(valid, result, self.badvalue)
        return self._out(result)

    def logpmf(self, k, *args, **kwds):
        with np.errstate(divide="ignore"):
            return np.log(self.pmf(k, *args, **kwds))

    def cdf(self, k, *args, **kwds):
        k0, shapes, loc, _scale = self._value_binder(self, k, *args, **kwds)
        k0 = np.asarray(k0, dtype=float)
        loc = np.asarray(loc, dtype=float)
        ks = np.floor(k0 - loc)
        a, b = self._get_support(*shapes)
        valid = self._valid(shapes, np.array(1.0))
        with np.errstate(all="ignore"):
            below = ks < a
            above = ks > b
            result = np.where(above, 1.0, np.where(below, 0.0, self._cdf(ks, *shapes)))
            result = np.where(valid, result, self.badvalue)
        return self._out(result)

    def logcdf(self, k, *args, **kwds):
        with np.errstate(divide="ignore"):
            return np.log(self.cdf(k, *args, **kwds))

    def sf(self, k, *args, **kwds):
        return 1.0 - np.asarray(self.cdf(k, *args, **kwds), dtype=float)

    def logsf(self, k, *args, **kwds):
        with np.errstate(divide="ignore"):
            return np.log(self.sf(k, *args, **kwds))

    def ppf(self, q, *args, **kwds):
        q0, shapes, loc, _scale = self._value_binder(self, q, *args, **kwds)
        q0 = np.asarray(q0, dtype=float)
        loc = np.asarray(loc, dtype=float)
        a, b = self._get_support(*shapes)
        valid = self._valid(shapes, np.array(1.0)) & (q0 >= 0.0) & (q0 <= 1.0)
        with np.errstate(all="ignore"):
            clipped = np.clip(q0, 0.0, 1.0)
            core = np.where(q0 <= 0.0, a, np.where(q0 >= 1.0, b, self._ppf(clipped, *shapes)))
            result = np.where(valid, loc + core, self.badvalue)
        return self._out(result)

    def isf(self, q, *args, **kwds):
        return self.ppf(1.0 - np.asarray(q, dtype=float), *args, **kwds)

    def rvs(self, *args, size=None, random_state=None, **kwds):
        loc = kwds.pop("loc", 0)
        if kwds or len(args) != self.numargs:
            raise TypeError(f"{self.name}.rvs() got an unexpected argument")
        rng = check_random_state(random_state)
        raw = self._rvs(rng, size, *args)
        return np.asarray(raw) + loc

    def expect(
        self,
        func=None,
        args=(),
        loc=0,
        lb=None,
        ub=None,
        conditional=False,
        maxcount=1000,
        tolerance=1e-10,
        chunksize=32,
    ):
        if func is None:
            func = lambda k: k  # noqa: E731
        shapes = tuple(args)
        a, b = self._get_support(*shapes)
        low = a if lb is None else max(lb, a)
        high = b if ub is None else min(ub, b)
        low = int(math.ceil(low))
        total = 0.0
        k = low
        count = 0
        zero_run = 0
        while k <= high and count < maxcount:
            p = float(self.pmf(k, *shapes, loc=loc))
            if p > 0.0:
                total += func(k) * p
                zero_run = 0
            else:
                zero_run += 1
                if zero_run > 10 and k > low:
                    break
            k += 1
            count += 1
        return total


class _binom_gen(rv_discrete):
    def _argcheck(self, n, p):
        n_arr = np.asarray(n, dtype=float)
        p_arr = np.asarray(p, dtype=float)
        return (n_arr >= 0) & (n_arr == np.floor(n_arr)) & (p_arr >= 0) & (p_arr <= 1)

    def _get_support(self, n, p):
        return 0, n

    def _pmf(self, k, n, p):
        return _binom_pmf(k, n, p)

    def _cdf(self, k, n, p):
        return _binom_cdf(k, n, p)

    def _ppf(self, q, n, p):
        return _binom_ppf(q, n, p)

    def _stats(self, n, p):
        mean = n * p
        var = n * p * (1.0 - p)
        skew = (1.0 - 2.0 * p) / np.sqrt(var)
        kurt = (1.0 - 6.0 * p * (1.0 - p)) / var
        return mean, var, skew, kurt

    def _rvs(self, rng, size, n, p):
        return rng.binomial(n, p, size=size)

    def isf(self, q, *args, **kwds):
        # `_binom_isf` is exact (ported from Boost); avoid the generic `ppf(1 - q)` round trip.
        q0, shapes, loc, _scale = self._value_binder(self, q, *args, **kwds)
        q0 = np.asarray(q0, dtype=float)
        loc = np.asarray(loc, dtype=float)
        valid = self._valid(shapes, np.array(1.0)) & (q0 >= 0.0) & (q0 <= 1.0)
        with np.errstate(all="ignore"):
            result = np.where(valid, loc + _binom_isf(np.clip(q0, 0.0, 1.0), *shapes), self.badvalue)
        return self._out(result)


class _poisson_gen(rv_discrete):
    def _pmf(self, k, mu):
        with np.errstate(divide="ignore"):
            return np.exp(special.xlogy(k, mu) - mu - special.gammaln(k + 1.0))

    def _cdf(self, k, mu):
        return special.gammaincc(np.floor(k) + 1.0, mu)

    def _stats(self, mu):
        return mu, mu, 1.0 / np.sqrt(mu), 1.0 / mu

    def _rvs(self, rng, size, mu):
        return rng.poisson(mu, size=size)


binom = _binom_gen(name="binom")
poisson = _poisson_gen(name="poisson")
