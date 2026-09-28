"""Plain distribution classes for shellsim's ``scipy.stats``: six continuous distributions
(``norm``, ``t``, ``chi2``, ``f``, ``uniform``, ``expon``) and two discrete ones (``binom``,
``poisson``).

Unlike SciPy's ``rv_continuous``/``rv_discrete``, there is no public subclassing framework, no
``inspect.signature`` shape discovery, and no ``exec``-generated argument binders: each
distribution is a small class with named ``_pdf``/``_cdf``/``_ppf`` (or ``_pmf``) hooks taking
its shape parameters as plain positional arguments, and a shared base class applies ``loc``
(and, for continuous distributions, ``scale``). A wrong number of shape arguments fails through
Python's own call mechanism (calling ``self._pdf(x, *shapes)`` with the wrong ``len(shapes)``)
with an ordinary ``TypeError``, rather than a hand-reproduced SciPy error message.

Every hook is closed-form, built from ``scipy.special``'s incomplete gamma/beta ufuncs (which
already broadcast like ufuncs), so there is no elementwise Python loop for the continuous side.
The two discrete quantile functions (``binom``/``poisson`` ``ppf``/``isf``) need an integer
search; ``_integer_search`` below runs that search vectorized, over the whole array at once with
a bounded number of iterations, rather than looping in Python over array elements.
"""

import numpy as np

from scipy import special

__all__ = ["norm", "t", "chi2", "f", "uniform", "expon", "binom", "poisson"]


def _check_random_state(seed):
    """Turn `seed` into a `numpy.random.Generator` for `rvs`, via `default_rng` (`None`, an
    `int`, or an existing `Generator`). Kept local (rather than imported from
    `scipy._lib._util`) since this module no longer depends on `scipy._lib`, and deliberately
    does not touch the legacy `RandomState`/`mtrand` API."""
    return np.random.default_rng(seed)


def _scalarize(value):
    value = np.asarray(value)
    return value[()] if value.ndim == 0 else value


# ------------------------------------------------------------------------------------------------
# Student's t, chi-square and F cdf/ppf: shared with scipy.stats._tests, which needs the same
# tail probabilities for its t-tests and chi-square tests.
# ------------------------------------------------------------------------------------------------


def _t_cdf(t, df):
    """Student's t CDF via `x = df / (df + t^2)` and the incomplete beta function
    (Abramowitz & Stegun 26.7.1)."""
    t = np.asarray(t, dtype=np.float64)
    df = np.asarray(df, dtype=np.float64)
    x = df / (df + t * t)
    half = special.betainc(df / 2.0, 0.5, x)
    return np.where(t >= 0.0, 1.0 - 0.5 * half, 0.5 * half)


def _t_ppf(p, df):
    """Solve `_t_cdf(t, df) = p` for `t`."""
    p = np.asarray(p, dtype=np.float64)
    df = np.asarray(df, dtype=np.float64)
    target = np.where(p < 0.5, 2.0 * p, 2.0 * (1.0 - p))
    x = special.betaincinv(df / 2.0, 0.5, target)
    magnitude = np.sqrt(df * (1.0 - x) / x)
    return np.where(p < 0.5, -magnitude, magnitude)


def _chi2_cdf(x, df):
    return special.gammainc(df / 2.0, x / 2.0)


def _chi2_ppf(q, df):
    return 2.0 * special.gammainccinv(df / 2.0, 1.0 - q)


def _f_cdf(x, dfn, dfd):
    z = dfn * x / (dfn * x + dfd)
    return special.betainc(dfn / 2.0, dfd / 2.0, z)


def _f_ppf(q, dfn, dfd):
    z = special.betaincinv(dfn / 2.0, dfd / 2.0, q)
    return dfd * z / (dfn * (1.0 - z))


# ------------------------------------------------------------------------------------------------
# Continuous distributions
# ------------------------------------------------------------------------------------------------


class _ContinuousDistribution:
    """Shared plumbing for the six continuous distributions: applies `loc`/`scale` to a
    subclass's standard-form `_pdf`/`_cdf`/`_ppf`/`_mean`/`_var`/`_rvs` hooks."""

    def __init__(self, name, numargs=0, a=-np.inf, b=np.inf):
        self.name = name
        self.numargs = numargs
        self.a = a
        self.b = b

    def __call__(self, *args, **kwds):
        return _FrozenContinuous(self, args, kwds)

    def __repr__(self):
        return f"<scipy.stats._distributions.{type(self).__name__} object>"

    def _bind(self, args, kwds):
        """Split `(shape args..., [loc[, scale]])` from `args`/`kwds`, the way SciPy's own
        `dist.pdf(x, *shapeargs, loc=0, scale=1)` calling convention allows `loc`/`scale`
        positionally or by keyword, but not both."""
        n = self.numargs
        if len(args) < n:
            raise TypeError(f"{self.name}() missing required shape parameter(s)")
        shapes, rest = args[:n], args[n:]
        if len(rest) > 2:
            raise TypeError(f"{self.name}() takes at most {n + 2} positional arguments")
        if len(rest) >= 1:
            if "loc" in kwds:
                raise TypeError(f"{self.name}() got multiple values for argument 'loc'")
            loc = rest[0]
        else:
            loc = kwds.pop("loc", 0.0)
        if len(rest) >= 2:
            if "scale" in kwds:
                raise TypeError(f"{self.name}() got multiple values for argument 'scale'")
            scale = rest[1]
        else:
            scale = kwds.pop("scale", 1.0)
        if kwds:
            raise TypeError(f"{self.name}() got an unexpected keyword argument {next(iter(kwds))!r}")
        return shapes, float(loc), float(scale)

    def _argcheck(self, *shapes):
        ok = np.array(True)
        for shape in shapes:
            ok = ok & (np.asarray(shape, dtype=float) > 0)
        return ok

    def _valid(self, shapes, scale):
        valid = np.asarray(scale, dtype=float) > 0
        if shapes:
            valid = valid & self._argcheck(*shapes)
        return valid

    def pdf(self, x, *args, **kwds):
        shapes, loc, scale = self._bind(args, kwds)
        x = np.asarray(x, dtype=float)
        safe_scale = scale if scale > 0 else 1.0
        xs = (x - loc) / safe_scale
        valid = self._valid(shapes, scale)
        with np.errstate(all="ignore"):
            inside = (xs >= self.a) & (xs <= self.b)
            density = np.where(inside, self._pdf(xs, *shapes) / safe_scale, 0.0)
            result = np.where(valid, density, np.nan)
        return _scalarize(result)

    def logpdf(self, x, *args, **kwds):
        with np.errstate(divide="ignore"):
            return np.log(self.pdf(x, *args, **kwds))

    def cdf(self, x, *args, **kwds):
        shapes, loc, scale = self._bind(args, kwds)
        x = np.asarray(x, dtype=float)
        safe_scale = scale if scale > 0 else 1.0
        xs = (x - loc) / safe_scale
        valid = self._valid(shapes, scale)
        with np.errstate(all="ignore"):
            result = np.where(xs > self.b, 1.0, np.where(xs < self.a, 0.0, self._cdf(xs, *shapes)))
            result = np.where(valid, result, np.nan)
        return _scalarize(result)

    def logcdf(self, x, *args, **kwds):
        with np.errstate(divide="ignore"):
            return np.log(self.cdf(x, *args, **kwds))

    def sf(self, x, *args, **kwds):
        return 1.0 - np.asarray(self.cdf(x, *args, **kwds), dtype=float)

    def logsf(self, x, *args, **kwds):
        with np.errstate(divide="ignore"):
            return np.log(self.sf(x, *args, **kwds))

    def ppf(self, q, *args, **kwds):
        shapes, loc, scale = self._bind(args, kwds)
        q = np.asarray(q, dtype=float)
        valid = self._valid(shapes, scale) & (q >= 0.0) & (q <= 1.0)
        with np.errstate(all="ignore"):
            clipped = np.clip(q, 0.0, 1.0)
            core = np.where(q <= 0.0, self.a, np.where(q >= 1.0, self.b, self._ppf(clipped, *shapes)))
            result = np.where(valid, loc + scale * core, np.nan)
        return _scalarize(result)

    def isf(self, q, *args, **kwds):
        return self.ppf(1.0 - np.asarray(q, dtype=float), *args, **kwds)

    def mean(self, *args, **kwds):
        shapes, loc, scale = self._bind(args, kwds)
        valid = self._valid(shapes, scale)
        value = loc + scale * np.asarray(self._mean(*shapes), dtype=float)
        return _scalarize(np.where(valid, value, np.nan))

    def var(self, *args, **kwds):
        shapes, loc, scale = self._bind(args, kwds)
        valid = self._valid(shapes, scale)
        value = scale**2 * np.asarray(self._var(*shapes), dtype=float)
        return _scalarize(np.where(valid, value, np.nan))

    def std(self, *args, **kwds):
        return _scalarize(np.sqrt(np.asarray(self.var(*args, **kwds), dtype=float)))

    def median(self, *args, **kwds):
        return self.ppf(0.5, *args, **kwds)

    def interval(self, confidence, *args, **kwds):
        alpha = (1.0 - np.asarray(confidence, dtype=float)) / 2.0
        return self.ppf(alpha, *args, **kwds), self.ppf(1.0 - alpha, *args, **kwds)

    def rvs(self, *args, size=None, random_state=None, **kwds):
        shapes, loc, scale = self._bind(args, kwds)
        rng = _check_random_state(random_state)
        raw = self._rvs(rng, size, *shapes)
        return loc + scale * np.asarray(raw)


class _FrozenContinuous:
    """A continuous distribution with its shape/`loc`/`scale` arguments fixed, as `dist(...)`
    returns."""

    _METHODS = (
        "pdf", "logpdf", "cdf", "logcdf", "sf", "logsf", "ppf", "isf",
        "mean", "var", "std", "median", "interval", "rvs",
    )

    def __init__(self, dist, args, kwds):
        self._dist = dist
        self._args = args
        self._kwds = kwds
        for name in self._METHODS:
            setattr(self, name, self._forward(getattr(dist, name)))

    def _forward(self, method):
        def call(*args, **kwds):
            merged = dict(self._kwds)
            merged.update(kwds)
            return method(*args, *self._args, **merged)

        return call


class _Norm(_ContinuousDistribution):
    def _pdf(self, x):
        return np.exp(-0.5 * x * x) / np.sqrt(2.0 * np.pi)

    def _cdf(self, x):
        return special.ndtr(x)

    def _ppf(self, q):
        return special.ndtri(q)

    def _mean(self):
        return 0.0

    def _var(self):
        return 1.0

    def _rvs(self, rng, size):
        return rng.standard_normal(size=size)


class _T(_ContinuousDistribution):
    def _pdf(self, x, df):
        return np.exp(
            -0.5 * (df + 1.0) * np.log1p(x * x / df) - 0.5 * np.log(df) - special.betaln(0.5, 0.5 * df)
        )

    def _cdf(self, x, df):
        return _t_cdf(x, df)

    def _ppf(self, q, df):
        return _t_ppf(q, df)

    def _mean(self, df):
        return np.where(df > 1, 0.0, np.nan)

    def _var(self, df):
        return np.where(df > 2, df / (df - 2.0), np.where(df > 1, np.inf, np.nan))

    def _rvs(self, rng, size, df):
        return rng.standard_t(df, size=size)


class _Chi2(_ContinuousDistribution):
    def _pdf(self, x, df):
        with np.errstate(divide="ignore", invalid="ignore"):
            logpdf = (0.5 * df - 1.0) * np.log(x) - 0.5 * x - 0.5 * df * np.log(2.0) - special.gammaln(0.5 * df)
        return np.exp(logpdf)

    def _cdf(self, x, df):
        return _chi2_cdf(x, df)

    def _ppf(self, q, df):
        return _chi2_ppf(q, df)

    def _mean(self, df):
        return df

    def _var(self, df):
        return 2.0 * df

    def _rvs(self, rng, size, df):
        return rng.chisquare(df, size=size)


class _F(_ContinuousDistribution):
    def _pdf(self, x, dfn, dfd):
        with np.errstate(divide="ignore", invalid="ignore"):
            logpdf = (
                0.5 * dfn * np.log(dfn)
                + 0.5 * dfd * np.log(dfd)
                + (0.5 * dfn - 1.0) * np.log(x)
                - 0.5 * (dfn + dfd) * np.log(dfd + dfn * x)
                - special.betaln(0.5 * dfn, 0.5 * dfd)
            )
        return np.exp(logpdf)

    def _cdf(self, x, dfn, dfd):
        return _f_cdf(x, dfn, dfd)

    def _ppf(self, q, dfn, dfd):
        return _f_ppf(q, dfn, dfd)

    def _mean(self, dfn, dfd):
        return np.where(dfd > 2, dfd / (dfd - 2.0), np.nan)

    def _var(self, dfn, dfd):
        return np.where(
            dfd > 4,
            2.0 * dfd**2 * (dfn + dfd - 2.0) / (dfn * (dfd - 2.0) ** 2 * (dfd - 4.0)),
            np.nan,
        )

    def _rvs(self, rng, size, dfn, dfd):
        return rng.f(dfn, dfd, size=size)


class _Uniform(_ContinuousDistribution):
    def _pdf(self, x):
        return np.ones_like(x)

    def _cdf(self, x):
        return x

    def _ppf(self, q):
        return q

    def _mean(self):
        return 0.5

    def _var(self):
        return 1.0 / 12.0

    def _rvs(self, rng, size):
        return rng.random(size=size)


class _Expon(_ContinuousDistribution):
    def _pdf(self, x):
        return np.exp(-x)

    def _cdf(self, x):
        return -np.expm1(-x)

    def _ppf(self, q):
        return -np.log1p(-q)

    def _mean(self):
        return 1.0

    def _var(self):
        return 1.0

    def _rvs(self, rng, size):
        return rng.standard_exponential(size=size)


norm = _Norm("norm")
t = _T("t", numargs=1)
chi2 = _Chi2("chi2", numargs=1, a=0.0)
f = _F("f", numargs=2, a=0.0)
uniform = _Uniform("uniform", a=0.0, b=1.0)
expon = _Expon("expon", a=0.0)


# ------------------------------------------------------------------------------------------------
# Discrete distributions
# ------------------------------------------------------------------------------------------------


def _integer_search(cdf, q, lo, hi):
    """Smallest integer `k` in `[lo, hi]` with `cdf(k) >= q`, by bisection. `cdf` is
    non-decreasing in `k` (a discrete CDF), so ordinary bisection applies; this runs it vectorized
    over the whole array at once (a bounded number of iterations, each a single call to `cdf`
    over every element), rather than searching one array element at a time in a Python loop.
    """
    lo, hi, q = np.broadcast_arrays(np.asarray(lo, dtype=np.float64), np.asarray(hi, dtype=np.float64), q)
    lo = lo.copy()
    hi = hi.copy()
    # A generous fixed iteration count keeps this simple; the CPU budget is not a design
    # constraint, and 64 doublings of the bisection interval is far more than any of `hi`'s
    # practical magnitudes need.
    for _ in range(64):
        active = lo < hi
        if not np.any(active):
            break
        mid = np.floor((lo + hi) / 2.0)
        below = active & (cdf(mid) < q)
        hi = np.where(active & ~below, mid, hi)
        lo = np.where(below, mid + 1.0, lo)
    return lo


def _poisson_upper_bound(mu, q):
    """A `k` with `poisson(mu).cdf(k) >= q`, generous enough to bracket `_integer_search`: a
    normal-approximation guess, doubled (vectorized, over the whole array) until it holds."""
    mu = np.asarray(mu, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    hi = np.maximum(np.ceil(mu + 10.0 * np.sqrt(mu + 1.0) + 10.0), 1.0)
    for _ in range(64):
        insufficient = special.gammaincc(hi + 1.0, mu) < q
        if not np.any(insufficient):
            break
        hi = np.where(insufficient, hi * 2.0, hi)
    return hi


class _DiscreteDistribution:
    """Shared plumbing for the two discrete distributions (`loc` only, no `scale`)."""

    def __init__(self, name, numargs, a=0.0, b=np.inf):
        self.name = name
        self.numargs = numargs
        self._a = a
        self._b = b

    def __call__(self, *args, **kwds):
        return _FrozenDiscrete(self, args, kwds)

    def __repr__(self):
        return f"<scipy.stats._distributions.{type(self).__name__} object>"

    def _get_support(self, *shapes):
        return self._a, self._b

    def _bind(self, args, kwds):
        n = self.numargs
        if len(args) < n:
            raise TypeError(f"{self.name}() missing required shape parameter(s)")
        shapes, rest = args[:n], args[n:]
        if len(rest) > 1:
            raise TypeError(f"{self.name}() takes at most {n + 1} positional arguments")
        if rest:
            if "loc" in kwds:
                raise TypeError(f"{self.name}() got multiple values for argument 'loc'")
            loc = rest[0]
        else:
            loc = kwds.pop("loc", 0.0)
        if kwds:
            raise TypeError(f"{self.name}() got an unexpected keyword argument {next(iter(kwds))!r}")
        return shapes, float(loc)

    def _argcheck(self, *shapes):
        ok = np.array(True)
        for shape in shapes:
            ok = ok & (np.asarray(shape, dtype=float) > 0)
        return ok

    def _valid(self, shapes):
        return self._argcheck(*shapes) if shapes else np.array(True)

    def pmf(self, k, *args, **kwds):
        shapes, loc = self._bind(args, kwds)
        a, b = self._get_support(*shapes)
        k = np.asarray(k, dtype=float)
        ks = k - loc
        valid = self._valid(shapes)
        with np.errstate(all="ignore"):
            inside = (ks >= a) & (ks <= b) & (ks == np.floor(ks))
            result = np.where(inside, self._pmf(ks, *shapes), 0.0)
            result = np.where(valid, result, np.nan)
        return _scalarize(result)

    def logpmf(self, k, *args, **kwds):
        with np.errstate(divide="ignore"):
            return np.log(self.pmf(k, *args, **kwds))

    def cdf(self, k, *args, **kwds):
        shapes, loc = self._bind(args, kwds)
        a, b = self._get_support(*shapes)
        k = np.asarray(k, dtype=float)
        ks = np.floor(k - loc)
        valid = self._valid(shapes)
        with np.errstate(all="ignore"):
            result = np.where(ks > b, 1.0, np.where(ks < a, 0.0, self._cdf(ks, *shapes)))
            result = np.where(valid, result, np.nan)
        return _scalarize(result)

    def logcdf(self, k, *args, **kwds):
        with np.errstate(divide="ignore"):
            return np.log(self.cdf(k, *args, **kwds))

    def sf(self, k, *args, **kwds):
        return 1.0 - np.asarray(self.cdf(k, *args, **kwds), dtype=float)

    def logsf(self, k, *args, **kwds):
        with np.errstate(divide="ignore"):
            return np.log(self.sf(k, *args, **kwds))

    def ppf(self, q, *args, **kwds):
        shapes, loc = self._bind(args, kwds)
        a, b = self._get_support(*shapes)
        q = np.asarray(q, dtype=float)
        valid = self._valid(shapes) & (q >= 0.0) & (q <= 1.0)
        with np.errstate(all="ignore"):
            clipped = np.clip(q, 0.0, 1.0)
            core = np.where(q <= 0.0, a, np.where(q >= 1.0, b, self._ppf(clipped, *shapes)))
            result = np.where(valid, loc + core, np.nan)
        return _scalarize(result)

    def isf(self, q, *args, **kwds):
        return self.ppf(1.0 - np.asarray(q, dtype=float), *args, **kwds)

    def mean(self, *args, **kwds):
        shapes, loc = self._bind(args, kwds)
        valid = self._valid(shapes)
        value = loc + np.asarray(self._mean(*shapes), dtype=float)
        return _scalarize(np.where(valid, value, np.nan))

    def var(self, *args, **kwds):
        shapes, loc = self._bind(args, kwds)
        valid = self._valid(shapes)
        value = np.asarray(self._var(*shapes), dtype=float)
        return _scalarize(np.where(valid, value, np.nan))

    def std(self, *args, **kwds):
        return _scalarize(np.sqrt(np.asarray(self.var(*args, **kwds), dtype=float)))

    def median(self, *args, **kwds):
        return self.ppf(0.5, *args, **kwds)

    def interval(self, confidence, *args, **kwds):
        alpha = (1.0 - np.asarray(confidence, dtype=float)) / 2.0
        return self.ppf(alpha, *args, **kwds), self.ppf(1.0 - alpha, *args, **kwds)

    def rvs(self, *args, size=None, random_state=None, **kwds):
        shapes, loc = self._bind(args, kwds)
        rng = _check_random_state(random_state)
        raw = self._rvs(rng, size, *shapes)
        return np.asarray(raw) + loc


class _FrozenDiscrete:
    """A discrete distribution with its shape/`loc` arguments fixed, as `dist(...)` returns."""

    _METHODS = (
        "pmf", "logpmf", "cdf", "logcdf", "sf", "logsf", "ppf", "isf",
        "mean", "var", "std", "median", "interval", "rvs",
    )

    def __init__(self, dist, args, kwds):
        self._dist = dist
        self._args = args
        self._kwds = kwds
        for name in self._METHODS:
            setattr(self, name, self._forward(getattr(dist, name)))

    def _forward(self, method):
        def call(*args, **kwds):
            merged = dict(self._kwds)
            merged.update(kwds)
            return method(*args, *self._args, **merged)

        return call


class _Binom(_DiscreteDistribution):
    def _get_support(self, n, p):
        return 0.0, n

    def _pmf(self, k, n, p):
        with np.errstate(divide="ignore"):
            log_coeff = special.gammaln(n + 1.0) - special.gammaln(k + 1.0) - special.gammaln(n - k + 1.0)
            return np.exp(log_coeff + special.xlogy(k, p) + special.xlogy(n - k, 1.0 - p))

    def _cdf(self, k, n, p):
        return np.where(k >= n, 1.0, special.betaincc(k + 1.0, n - k, p))

    def _ppf(self, q, n, p):
        return _integer_search(lambda k: self._cdf(k, n, p), q, 0.0, n)

    def _mean(self, n, p):
        return n * p

    def _var(self, n, p):
        return n * p * (1.0 - p)

    def _rvs(self, rng, size, n, p):
        return rng.binomial(n, p, size=size)


class _Poisson(_DiscreteDistribution):
    def _pmf(self, k, mu):
        with np.errstate(divide="ignore"):
            return np.exp(special.xlogy(k, mu) - mu - special.gammaln(k + 1.0))

    def _cdf(self, k, mu):
        return special.gammaincc(np.floor(k) + 1.0, mu)

    def _ppf(self, q, mu):
        hi = _poisson_upper_bound(mu, q)
        return _integer_search(lambda k: self._cdf(k, mu), q, 0.0, hi)

    def _mean(self, mu):
        return mu

    def _var(self, mu):
        return mu

    def _rvs(self, rng, size, mu):
        return rng.poisson(mu, size=size)


binom = _Binom("binom", numargs=2)
poisson = _Poisson("poisson", numargs=1)
