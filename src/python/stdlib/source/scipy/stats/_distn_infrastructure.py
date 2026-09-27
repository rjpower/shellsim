"""The ``rv_continuous`` and ``rv_discrete`` machinery, following SciPy 1.18's
``scipy/stats/_distn_infrastructure.py``.

The public methods validate and broadcast arguments, evaluate the private ``_pdf``, ``_cdf`` and
similar methods of a distribution on the valid elements, and fill the rest with ``badvalue``,
as SciPy's do. Two things differ:

- SciPy builds ``_parse_args`` and its variants with ``exec`` from the shape parameters it
  reads off the signatures of ``_pdf`` and ``_cdf`` (``_pmf`` and ``_cdf`` for discrete
  distributions). shellsim cannot ``exec``, so the binder here reproduces those generated
  functions, including CPython's ``TypeError`` messages. The shapes are found as SciPy finds
  them, through ``getfullargspec_no_self``.
- ``rv_discrete(values=...)`` needs ``__new__`` to return an ``rv_sample``, which shellsim's
  object model does not honor, so it is rejected. There are no docstrings or pickling hooks.

Methods that SciPy computes by numerical integration or root finding import ``scipy.integrate``
or ``scipy.optimize`` when called, so they fail explicitly while those modules are unsupported.
"""

import re
import warnings

import numpy as np
from numpy import (
    arange,
    asarray,
    empty,
    floor,
    inf,
    isinf,
    log,
    logical_and,
    nan,
    ndarray,
    ones,
    place,
    putmask,
    shape,
    sqrt,
    vectorize,
    zeros,
)
from scipy._lib._util import apply_where, check_random_state
from scipy._lib._util import getfullargspec_no_self as _getfullargspec
from scipy.special import comb, entr
from scipy.stats._finite_differences import _derivative

_XMAX = np.finfo(float).max
_LOGXMAX = np.log(_XMAX)

# Python keywords, which cannot name shape parameters (``keyword.kwlist``).
_KEYWORDS = frozenset(
    (
        "False None True and as assert async await break class continue def del elif else "
        "except finally for from global if import in is lambda nonlocal not or pass raise "
        "return try while with yield"
    ).split()
)


def _moment_from_stats(n, mu, mu2, g1, g2, moment_func, args):
    """The ``n``-th non-central moment from the mean, variance, skewness and kurtosis."""
    if n == 0:
        return 1.0
    elif n == 1:
        if mu is None:
            val = moment_func(1, *args)
        else:
            val = mu
    elif n == 2:
        if mu2 is None or mu is None:
            val = moment_func(2, *args)
        else:
            val = mu2 + mu * mu
    elif n == 3:
        if g1 is None or mu2 is None or mu is None:
            val = moment_func(3, *args)
        else:
            mu3 = g1 * np.power(mu2, 1.5)  # 3rd central moment
            val = mu3 + 3 * mu * mu2 + mu * mu * mu  # 3rd non-central moment
    elif n == 4:
        if g1 is None or g2 is None or mu2 is None or mu is None:
            val = moment_func(4, *args)
        else:
            mu4 = (g2 + 3.0) * (mu2**2.0)  # 4th central moment
            mu3 = g1 * np.power(mu2, 1.5)  # 3rd central moment
            val = mu4 + 4 * mu * mu3 + 6 * mu * mu * mu2 + mu * mu * mu * mu
    else:
        val = moment_func(n, *args)
    return val


def _isintegral(x):
    return x == np.round(x)


def _sum_finite(x):
    """The sum of the finite elements of 1-D ``x`` and the number of non-finite ones."""
    finite_x = np.isfinite(x)
    bad_count = finite_x.size - np.count_nonzero(finite_x)
    return np.sum(x[finite_x]), bad_count


def _missing_names(names):
    """Quote and join argument names as CPython's "missing arguments" message does."""
    quoted = [f"'{name}'" for name in names]
    if len(quoted) == 1:
        return quoted[0]
    if len(quoted) == 2:
        return f"{quoted[0]} and {quoted[1]}"
    return ", ".join(quoted[:-1]) + ", and " + quoted[-1]


def _bind_arguments(name, required, optional, args, kwds):
    """Bind ``args`` and ``kwds`` as ``def name(self, *required, **optional)`` would.

    ``required`` are parameter names without defaults; ``optional`` are ``(name, default)``
    pairs. All are positional-or-keyword, and errors follow CPython's order and messages, with
    ``self`` counted among the positional arguments. Returns a dict from name to value.
    """
    names = required + tuple(optional_name for optional_name, _ in optional)
    values = dict(zip(names, args))
    for key, value in kwds.items():
        if key not in names:
            raise TypeError(f"{name}() got an unexpected keyword argument '{key}'")
        if key in values:
            raise TypeError(f"{name}() got multiple values for argument '{key}'")
        values[key] = value
    if len(args) > len(names):
        given = len(args) + 1
        raise TypeError(
            f"{name}() takes from {len(required) + 1} to {len(names) + 1} positional "
            f"arguments but {given} {'was' if given == 1 else 'were'} given"
        )
    missing = [required_name for required_name in required if required_name not in values]
    if missing:
        plural = "s" if len(missing) > 1 else ""
        raise TypeError(
            f"{name}() missing {len(missing)} required positional argument{plural}: "
            f"{_missing_names(missing)}"
        )
    for optional_name, default in optional:
        if optional_name not in values:
            values[optional_name] = default
    return values


# Frozen RV class
class rv_frozen:
    def __init__(self, dist, *args, **kwds):
        self.args = args
        self.kwds = kwds

        # create a new instance
        self.dist = dist.__class__(**dist._updated_ctor_param())

        shapes, _, _ = self.dist._parse_args(*args, **kwds)
        self.a, self.b = self.dist._get_support(*shapes)

    @property
    def random_state(self):
        return self.dist._random_state

    @random_state.setter
    def random_state(self, seed):
        self.dist._random_state = check_random_state(seed)

    def cdf(self, x):
        return self.dist.cdf(x, *self.args, **self.kwds)

    def logcdf(self, x):
        return self.dist.logcdf(x, *self.args, **self.kwds)

    def ppf(self, q):
        return self.dist.ppf(q, *self.args, **self.kwds)

    def isf(self, q):
        return self.dist.isf(q, *self.args, **self.kwds)

    def rvs(self, size=None, random_state=None):
        kwds = self.kwds.copy()
        kwds.update({"size": size, "random_state": random_state})
        return self.dist.rvs(*self.args, **kwds)

    def sf(self, x):
        return self.dist.sf(x, *self.args, **self.kwds)

    def logsf(self, x):
        return self.dist.logsf(x, *self.args, **self.kwds)

    def stats(self, moments="mv"):
        kwds = self.kwds.copy()
        kwds.update({"moments": moments})
        return self.dist.stats(*self.args, **kwds)

    def median(self):
        return self.dist.median(*self.args, **self.kwds)

    def mean(self):
        return self.dist.mean(*self.args, **self.kwds)

    def var(self):
        return self.dist.var(*self.args, **self.kwds)

    def std(self):
        return self.dist.std(*self.args, **self.kwds)

    def moment(self, order=None):
        return self.dist.moment(order, *self.args, **self.kwds)

    def entropy(self):
        return self.dist.entropy(*self.args, **self.kwds)

    def interval(self, confidence=None):
        return self.dist.interval(confidence, *self.args, **self.kwds)

    def expect(self, func=None, lb=None, ub=None, conditional=False, **kwds):
        # expect only accepts shape parameters positionally, so convert args, kwds, loc
        # and scale.
        a, loc, scale = self.dist._parse_args(*self.args, **self.kwds)
        if isinstance(self.dist, rv_discrete):
            return self.dist.expect(func, a, loc, lb, ub, conditional, **kwds)
        else:
            return self.dist.expect(func, a, loc, scale, lb, ub, conditional, **kwds)

    def support(self):
        return self.dist.support(*self.args, **self.kwds)


class rv_discrete_frozen(rv_frozen):
    def pmf(self, k):
        return self.dist.pmf(k, *self.args, **self.kwds)

    def logpmf(self, k):  # No error
        return self.dist.logpmf(k, *self.args, **self.kwds)


class rv_continuous_frozen(rv_frozen):
    def pdf(self, x):
        return self.dist.pdf(x, *self.args, **self.kwds)

    def logpdf(self, x):
        return self.dist.logpdf(x, *self.args, **self.kwds)


def argsreduce(cond, *args):
    """Broadcast ``args`` with ``cond`` and keep the elements where ``cond`` holds, in 1-D.

    Arguments of size one are returned as 1-element arrays rather than repeated.
    """
    # some distributions assume arguments are iterable.
    newargs = np.atleast_1d(*args)

    # np.atleast_1d returns an array if only one argument, or a list of arrays
    # if more than one argument.
    if not isinstance(newargs, (list, tuple)):
        newargs = (newargs,)

    if np.all(cond):
        # broadcast arrays with cond
        *newargs, cond = np.broadcast_arrays(*newargs, cond)
        return [arg.ravel() for arg in newargs]

    s = cond.shape
    # np.extract returns flattened arrays, which are not broadcastable together
    # unless they are either the same size or size == 1.
    return [
        (arg if np.size(arg) == 1 else np.extract(cond, np.broadcast_to(arg, s)))
        for arg in newargs
    ]


class rv_generic:
    """Class which encapsulates common functionality between rv_discrete and rv_continuous."""

    def __init__(self, seed=None):
        # figure out if _stats signature has 'moments' keyword
        sig = _getfullargspec(self._stats)
        self._stats_has_moments = (
            (sig.varkw is not None) or ("moments" in sig.args) or ("moments" in sig.kwonlyargs)
        )
        self._random_state = check_random_state(seed)

    @property
    def random_state(self):
        """Get or set the generator object for generating random variates."""
        return self._random_state

    @random_state.setter
    def random_state(self, seed):
        self._random_state = check_random_state(seed)

    def _construct_argparser(self, meths_to_inspect, locscale):
        """Record the shape parameters and the ``loc``/``scale`` parameters the parsers bind.

        ``locscale`` holds ``(name, default)`` pairs. If ``self.shapes`` is a non-empty string,
        it is a comma-separated list of shape parameters. Otherwise the shapes are the
        parameters after ``x`` in the signatures of ``meths_to_inspect``. Sets ``shapes`` and
        ``numargs``.
        """
        if self.shapes:
            # sanitize the user-supplied shapes
            if not isinstance(self.shapes, str):
                raise TypeError("shapes must be a string.")

            shapes = self.shapes.replace(",", " ").split()

            for field in shapes:
                if field in _KEYWORDS:
                    raise SyntaxError("keywords cannot be used as shapes.")
                if not re.match("^[_a-zA-Z][_a-zA-Z0-9]*$", field):
                    raise SyntaxError("shapes must be valid python identifiers")
        else:
            # find out the call signatures (_pdf, _cdf etc), deduce shape
            # arguments. Generic methods only have 'self, x', any further args
            # are shapes.
            shapes_list = []
            for meth in meths_to_inspect:
                shapes_args = _getfullargspec(meth)  # NB does not contain self
                args = shapes_args.args[1:]  # peel off 'x', too

                if args:
                    shapes_list.append(args)

                    # *args or **kwargs are not allowed w/automatic shapes
                    if shapes_args.varargs is not None:
                        raise TypeError("*args are not allowed w/out explicit shapes")
                    if shapes_args.varkw is not None:
                        raise TypeError("**kwds are not allowed w/out explicit shapes")
                    if shapes_args.kwonlyargs:
                        raise TypeError("kwonly args are not allowed w/out explicit shapes")
                    if shapes_args.defaults is not None:
                        raise TypeError("defaults are not allowed for shapes")

            if shapes_list:
                shapes = shapes_list[0]

                # make sure the signatures are consistent
                for item in shapes_list:
                    if item != shapes:
                        raise TypeError("Shape arguments are inconsistent.")
            else:
                shapes = []

        self._shape_list = tuple(shapes)
        self._locscale = tuple(locscale)
        self.shapes = ", ".join(shapes) if shapes else None
        if not hasattr(self, "numargs"):
            # allows more general subclassing with *args
            self.numargs = len(shapes)

    def _bind(self, name, extra, args, kwds):
        values = _bind_arguments(name, self._shape_list, self._locscale + extra, args, kwds)
        shapes = tuple(values[shape_name] for shape_name in self._shape_list)
        scale = values["scale"] if "scale" in values else 1
        return shapes, values["loc"], scale, values

    def _parse_args(self, *args, **kwds):
        shapes, loc, scale, _ = self._bind("_parse_args", (), args, kwds)
        return shapes, loc, scale

    def _parse_args_rvs(self, *args, **kwds):
        shapes, loc, scale, values = self._bind("_parse_args_rvs", (("size", None),), args, kwds)
        return self._argcheck_rvs(*shapes, loc, scale, size=values["size"])

    def _parse_args_stats(self, *args, **kwds):
        shapes, loc, scale, values = self._bind(
            "_parse_args_stats", (("moments", "mv"),), args, kwds
        )
        return shapes, loc, scale, values["moments"]

    def freeze(self, *args, **kwds):
        """Freeze the distribution for the given arguments."""
        if isinstance(self, rv_continuous):
            return rv_continuous_frozen(self, *args, **kwds)
        else:
            return rv_discrete_frozen(self, *args, **kwds)

    def __call__(self, *args, **kwds):
        return self.freeze(*args, **kwds)

    # The actual calculation functions (no basic checking need be done)
    # If these are defined, the others won't be looked at.
    # Otherwise, the other set can be defined.
    def _stats(self, *args, **kwds):
        return None, None, None, None

    # Noncentral moments (also known as the moment about the origin).
    def _munp(self, n, *args):
        # Silence floating point warnings from integration.
        with np.errstate(all="ignore"):
            vals = self.generic_moment(n, *args)
        return vals

    def _argcheck_rvs(self, *args, **kwargs):
        # Handle broadcasting and size validation of the rvs method.
        # `args` holds the shape parameters, the location and the scale, in that order. If
        # `size` is not None, it gives the shape of the result.
        size = kwargs.get("size", None)
        all_bcast = np.broadcast_arrays(*args)

        def squeeze_left(a):
            while a.ndim > 0 and a.shape[0] == 1:
                a = a[0]
            return a

        # Eliminate trivial leading dimensions, as NumPy's random variate generators
        # effectively ignore them when `size` is given.
        all_bcast = [squeeze_left(a) for a in all_bcast]
        bcast_shape = all_bcast[0].shape
        bcast_ndim = all_bcast[0].ndim

        if size is None:
            size_ = bcast_shape
        else:
            size_ = tuple(np.atleast_1d(size))

        # Check compatibility of size_ with the broadcast shape of all the parameters: a
        # dimension of size_ must equal the parameters' dimension or the latter must be 1.
        ndiff = bcast_ndim - len(size_)
        if ndiff < 0:
            bcast_shape = (1,) * (-ndiff) + bcast_shape
        elif ndiff > 0:
            size_ = (1,) * ndiff + size_

        ok = all([bcdim == 1 or bcdim == szdim for (bcdim, szdim) in zip(bcast_shape, size_)])
        if not ok:
            raise ValueError(
                "size does not match the broadcast shape of "
                f"the parameters. {size}, {size_}, {bcast_shape}"
            )

        param_bcast = all_bcast[:-2]
        loc_bcast = all_bcast[-2]
        scale_bcast = all_bcast[-1]

        return param_bcast, loc_bcast, scale_bcast, size_

    # These are the methods you must define (standard form functions)
    # NB: generic _pdf, _logpdf, _cdf are different for
    # rv_continuous and rv_discrete hence are defined in there
    def _argcheck(self, *args):
        """Default check for correct values on args and keywords.

        Returns condition array of 1's where arguments are correct and
         0's where they are not.
        """
        cond = 1
        for arg in args:
            cond = logical_and(cond, (asarray(arg) > 0))
        return cond

    def _get_support(self, *args, **kwargs):
        """Return the support of the (unscaled, unshifted) distribution."""
        return self.a, self.b

    def _support_mask(self, x, *args):
        a, b = self._get_support(*args)
        with np.errstate(invalid="ignore"):
            return (a <= x) & (x <= b)

    def _open_support_mask(self, x, *args):
        a, b = self._get_support(*args)
        with np.errstate(invalid="ignore"):
            return (a < x) & (x < b)

    def _rvs(self, *args, size=None, random_state=None):
        # Use basic inverse cdf algorithm for RV generation as default.
        U = random_state.uniform(size=size)
        Y = self._ppf(U, *args)
        return Y

    def _logcdf(self, x, *args):
        with np.errstate(divide="ignore"):
            return log(self._cdf(x, *args))

    def _sf(self, x, *args):
        return 1.0 - self._cdf(x, *args)

    def _logsf(self, x, *args):
        with np.errstate(divide="ignore"):
            return log(self._sf(x, *args))

    def _ppf(self, q, *args):
        return self._ppfvec(q, *args)

    def _isf(self, q, *args):
        return self._ppf(1.0 - q, *args)  # use correct _ppf for subclasses

    # These are actually called, and should not be overwritten if you
    # want to keep error checking.
    def rvs(self, *args, **kwds):
        """Random variates of given type."""
        discrete = kwds.pop("discrete", None)
        rndm = kwds.pop("random_state", None)
        args, loc, scale, size = self._parse_args_rvs(*args, **kwds)
        cond = logical_and(self._argcheck(*args), (scale >= 0))
        if not np.all(cond):
            message = (
                "Domain error in arguments. The `scale` parameter must "
                "be positive for all distributions, and many "
                "distributions have restrictions on shape parameters. "
                f"Please see the `scipy.stats.{self.name}` "
                "documentation for details."
            )
            raise ValueError(message)

        if np.all(scale == 0):
            return loc * ones(size, "d")

        # extra gymnastics needed for a custom random_state
        if rndm is not None:
            random_state_saved = self._random_state
            random_state = check_random_state(rndm)
        else:
            random_state = self._random_state

        vals = self._rvs(*args, size=size, random_state=random_state)

        vals = vals * scale + loc

        # do not forget to restore the _random_state
        if rndm is not None:
            self._random_state = random_state_saved

        # Cast to int if discrete
        if discrete:
            if size == ():
                vals = int(vals)
            else:
                vals = vals.astype(np.int64)

        return vals

    def stats(self, *args, **kwds):
        """Some statistics of the given RV: any of the mean, variance, skew and kurtosis."""
        args, loc, scale, moments = self._parse_args_stats(*args, **kwds)
        # scale = 1 by construction for discrete RVs
        loc, scale = map(asarray, (loc, scale))
        args = tuple(map(asarray, args))
        cond = self._argcheck(*args) & (scale > 0) & (loc == loc)
        output = []
        default = np.full(shape(cond), fill_value=self.badvalue)

        # Use only entries that are valid in calculation
        if np.any(cond):
            goodargs = argsreduce(cond, *(args + (scale, loc)))
            scale, loc, goodargs = goodargs[-2], goodargs[-1], goodargs[:-2]

            if self._stats_has_moments:
                mu, mu2, g1, g2 = self._stats(*goodargs, **{"moments": moments})
            else:
                mu, mu2, g1, g2 = self._stats(*goodargs)

            if "m" in moments:
                if mu is None:
                    mu = self._munp(1, *goodargs)
                out0 = default.copy()
                place(out0, cond, mu * scale + loc)
                output.append(out0)

            if "v" in moments:
                if mu2 is None:
                    mu2p = self._munp(2, *goodargs)
                    if mu is None:
                        mu = self._munp(1, *goodargs)
                    # if mean is inf then var is also inf
                    with np.errstate(invalid="ignore"):
                        mu2 = np.where(~np.isinf(mu), mu2p - mu**2, np.inf)
                out0 = default.copy()
                place(out0, cond, mu2 * scale * scale)
                output.append(out0)

            if "s" in moments:
                if g1 is None:
                    mu3p = self._munp(3, *goodargs)
                    if mu is None:
                        mu = self._munp(1, *goodargs)
                    if mu2 is None:
                        mu2p = self._munp(2, *goodargs)
                        with np.errstate(invalid="ignore"):
                            mu2 = mu2p - mu * mu
                    with np.errstate(invalid="ignore"):
                        mu3 = (-mu * mu - 3 * mu2) * mu + mu3p
                        g1 = mu3 / np.power(mu2, 1.5)
                out0 = default.copy()
                place(out0, cond, g1)
                output.append(out0)

            if "k" in moments:
                if g2 is None:
                    mu4p = self._munp(4, *goodargs)
                    if mu is None:
                        mu = self._munp(1, *goodargs)
                    if mu2 is None:
                        mu2p = self._munp(2, *goodargs)
                        with np.errstate(invalid="ignore"):
                            mu2 = mu2p - mu * mu
                    if g1 is None:
                        mu3 = None
                    else:
                        # (mu2**1.5) breaks down for nan and inf
                        mu3 = g1 * np.power(mu2, 1.5)
                    if mu3 is None:
                        mu3p = self._munp(3, *goodargs)
                        with np.errstate(invalid="ignore"):
                            mu3 = (-mu * mu - 3 * mu2) * mu + mu3p
                    with np.errstate(invalid="ignore"):
                        mu4 = ((-(mu**2) - 6 * mu2) * mu - 4 * mu3) * mu + mu4p
                        g2 = mu4 / mu2**2.0 - 3.0
                out0 = default.copy()
                place(out0, cond, g2)
                output.append(out0)
        else:  # no valid args
            output = [default.copy() for _ in moments]

        output = [out[()] for out in output]
        if len(output) == 1:
            return output[0]
        else:
            return tuple(output)

    def entropy(self, *args, **kwds):
        """Differential entropy of the RV."""
        args, loc, scale = self._parse_args(*args, **kwds)
        # NB: for discrete distributions scale=1 by construction in _parse_args
        loc, scale = map(asarray, (loc, scale))
        args = tuple(map(asarray, args))
        cond0 = self._argcheck(*args) & (scale > 0) & (loc == loc)
        output = zeros(shape(cond0), "d")
        place(output, (1 - cond0), self.badvalue)
        goodargs = argsreduce(cond0, scale, *args)
        goodscale = goodargs[0]
        goodargs = goodargs[1:]
        place(output, cond0, self.vecentropy(*goodargs) + log(goodscale))
        return output[()]

    def moment(self, order, *args, **kwds):
        """Non-central moment of the specified order."""
        n = order
        shapes, loc, scale = self._parse_args(*args, **kwds)
        args = np.broadcast_arrays(*(*shapes, loc, scale))
        *shapes, loc, scale = args

        i0 = np.logical_and(self._argcheck(*shapes), scale > 0)
        i1 = np.logical_and(i0, loc == 0)
        i2 = np.logical_and(i0, loc != 0)

        args = argsreduce(i0, *shapes, loc, scale)
        *shapes, loc, scale = args

        if floor(n) != n:
            raise ValueError("Moment must be an integer.")
        if n < 0:
            raise ValueError("Moment must be positive.")
        mu, mu2, g1, g2 = None, None, None, None
        if (n > 0) and (n < 5):
            if self._stats_has_moments:
                mdict = {"moments": {1: "m", 2: "v", 3: "vs", 4: "mvsk"}[n]}
            else:
                mdict = {}
            mu, mu2, g1, g2 = self._stats(*shapes, **mdict)
        val = np.empty(loc.shape)  # val needs to be indexed by loc
        val[...] = _moment_from_stats(n, mu, mu2, g1, g2, self._munp, shapes)

        # Convert to transformed  X = L + S*Y
        # E[X^n] = E[(L+S*Y)^n] = L^n sum(comb(n, k)*(S/L)^k E[Y^k], k=0...n)
        result = zeros(i0.shape)
        place(result, ~i0, self.badvalue)

        if i1.any():
            res1 = scale[loc == 0] ** n * val[loc == 0]
            place(result, i1, res1)

        if i2.any():
            mom = [mu, mu2, g1, g2]
            arrs = [i for i in mom if i is not None]
            idx = [i for i in range(4) if mom[i] is not None]
            if any(idx):
                arrs = argsreduce(loc != 0, *arrs)
                j = 0
                for i in idx:
                    mom[i] = arrs[j]
                    j += 1
            mu, mu2, g1, g2 = mom
            args = argsreduce(loc != 0, *shapes, loc, scale, val)
            *shapes, loc, scale, val = args

            res2 = zeros(loc.shape, dtype="d")
            fac = scale / loc
            for k in range(n):
                valk = _moment_from_stats(k, mu, mu2, g1, g2, self._munp, shapes)
                res2 += comb(n, k, exact=True) * fac**k * valk
            res2 += fac**n * val
            res2 *= loc**n
            place(result, i2, res2)

        return result[()]

    def median(self, *args, **kwds):
        """Median of the distribution."""
        return self.ppf(0.5, *args, **kwds)

    def mean(self, *args, **kwds):
        """Mean of the distribution."""
        kwds["moments"] = "m"
        res = self.stats(*args, **kwds)
        if isinstance(res, ndarray) and res.ndim == 0:
            return res[()]
        return res

    def var(self, *args, **kwds):
        """Variance of the distribution."""
        kwds["moments"] = "v"
        res = self.stats(*args, **kwds)
        if isinstance(res, ndarray) and res.ndim == 0:
            return res[()]
        return res

    def std(self, *args, **kwds):
        """Standard deviation of the distribution."""
        kwds["moments"] = "v"
        res = sqrt(self.stats(*args, **kwds))
        return res

    def interval(self, confidence, *args, **kwds):
        """Confidence interval with equal areas around the median."""
        alpha = confidence

        alpha = asarray(alpha)
        if np.any((alpha > 1) | (alpha < 0)):
            raise ValueError("alpha must be between 0 and 1 inclusive")
        q1 = (1.0 - alpha) / 2
        q2 = (1.0 + alpha) / 2
        a = self.ppf(q1, *args, **kwds)
        b = self.ppf(q2, *args, **kwds)
        return a, b

    def support(self, *args, **kwargs):
        """Support of the distribution."""
        args, loc, scale = self._parse_args(*args, **kwargs)
        arrs = np.broadcast_arrays(*args, loc, scale)
        args, loc, scale = arrs[:-2], arrs[-2], arrs[-1]
        cond = self._argcheck(*args) & (scale > 0)
        _a, _b = self._get_support(*args)
        if cond.all():
            return _a * scale + loc, _b * scale + loc
        elif cond.ndim == 0:
            return self.badvalue, self.badvalue
        # promote bounds to at least float to fill in the badvalue
        _a, _b = np.asarray(_a).astype("d"), np.asarray(_b).astype("d")
        out_a, out_b = _a * scale + loc, _b * scale + loc
        place(out_a, 1 - cond, self.badvalue)
        place(out_b, 1 - cond, self.badvalue)
        return out_a, out_b

    def nnlf(self, theta, x):
        """Negative loglikelihood function."""
        loc, scale, args = self._unpack_loc_scale(theta)
        if not self._argcheck(*args) or scale <= 0:
            return inf
        x = (asarray(x) - loc) / scale
        n_log_scale = len(x) * log(scale)
        if np.any(~self._support_mask(x, *args)):
            return inf
        return self._nnlf(x, *args) + n_log_scale

    def _nnlf(self, x, *args):
        return -np.sum(self._logpxf(x, *args), axis=0)


class rv_continuous(rv_generic):
    """A generic continuous random variable class meant for subclassing."""

    def __init__(
        self,
        momtype=1,
        a=None,
        b=None,
        xtol=1e-14,
        badvalue=None,
        name=None,
        longname=None,
        shapes=None,
        seed=None,
    ):
        super().__init__(seed)

        # save the ctor parameters, cf generic freeze
        self._ctor_param = dict(
            momtype=momtype,
            a=a,
            b=b,
            xtol=xtol,
            badvalue=badvalue,
            name=name,
            longname=longname,
            shapes=shapes,
            seed=seed,
        )

        if badvalue is None:
            badvalue = nan
        if name is None:
            name = "Distribution"
        self.badvalue = badvalue
        self.name = name
        self.a = a
        self.b = b
        if a is None:
            self.a = -inf
        if b is None:
            self.b = inf
        self.xtol = xtol
        self.moment_type = momtype
        self.shapes = shapes

        self._construct_argparser(
            meths_to_inspect=[self._pdf, self._cdf], locscale=(("loc", 0), ("scale", 1))
        )
        self._attach_methods()

    def _attach_methods(self):
        self._ppfvec = vectorize(self._ppf_single, otypes="d")
        self._ppfvec.nin = self.numargs + 1
        self.vecentropy = vectorize(self._entropy, otypes="d")
        self._cdfvec = vectorize(self._cdf_single, otypes="d")
        self._cdfvec.nin = self.numargs + 1

        if self.moment_type == 0:
            self.generic_moment = vectorize(self._mom0_sc, otypes="d")
        else:
            self.generic_moment = vectorize(self._mom1_sc, otypes="d")
        self.generic_moment.nin = self.numargs + 1

    def _updated_ctor_param(self):
        dct = self._ctor_param.copy()
        dct["a"] = self.a
        dct["b"] = self.b
        dct["xtol"] = self.xtol
        dct["badvalue"] = self.badvalue
        dct["name"] = self.name
        dct["shapes"] = self.shapes
        return dct

    def _ppf_to_solve(self, x, q, *args):
        return self.cdf(*(x,) + args) - q

    def _ppf_single(self, q, *args):
        from scipy import optimize

        factor = 10.0
        left, right = self._get_support(*args)

        if np.isinf(left):
            left = min(-factor, right)
            while self._ppf_to_solve(left, q, *args) > 0.0:
                left, right = left * factor, left
            # left is now such that cdf(left) <= q
            # if right has changed, then cdf(right) > q

        if np.isinf(right):
            right = max(factor, left)
            while self._ppf_to_solve(right, q, *args) < 0.0:
                left, right = right, right * factor
            # right is now such that cdf(right) >= q

        return optimize.brentq(self._ppf_to_solve, left, right, args=(q,) + args, xtol=self.xtol)

    # moment from definition
    def _mom_integ0(self, x, m, *args):
        return x**m * self.pdf(x, *args)

    def _mom0_sc(self, m, *args):
        from scipy import integrate

        _a, _b = self._get_support(*args)
        return integrate.quad(self._mom_integ0, _a, _b, args=(m,) + args)[0]

    # moment calculated using ppf
    def _mom_integ1(self, q, m, *args):
        return (self.ppf(q, *args)) ** m

    def _mom1_sc(self, m, *args):
        from scipy import integrate

        return integrate.quad(self._mom_integ1, 0, 1, args=(m,) + args)[0]

    def _pdf(self, x, *args):
        return _derivative(self._cdf, x, dx=1e-5, args=args, order=5)

    # Could also define any of these
    def _logpdf(self, x, *args):
        p = self._pdf(x, *args)
        with np.errstate(divide="ignore"):
            return log(p)

    def _logpxf(self, x, *args):
        # continuous distributions have PDF, discrete have PMF, but sometimes
        # the distinction doesn't matter.
        return self._logpdf(x, *args)

    def _cdf_single(self, x, *args):
        from scipy import integrate

        _a, _b = self._get_support(*args)
        return integrate.quad(self._pdf, _a, x, args=args)[0]

    def _cdf(self, x, *args):
        return self._cdfvec(x, *args)

    def _logcdf(self, x, *args):
        median = self._ppf(0.5, *args)
        with np.errstate(divide="ignore"):
            return apply_where(
                x < median,
                (x,) + args,
                lambda x, *args: np.log(self._cdf(x, *args)),
                lambda x, *args: np.log1p(-self._sf(x, *args)),
            )

    def _logsf(self, x, *args):
        median = self._ppf(0.5, *args)
        with np.errstate(divide="ignore"):
            return apply_where(
                x > median,
                (x,) + args,
                lambda x, *args: np.log(self._sf(x, *args)),
                lambda x, *args: np.log1p(-self._cdf(x, *args)),
            )

    # generic _argcheck, _sf, _ppf, _isf, _rvs are defined
    # in rv_generic

    def pdf(self, x, *args, **kwds):
        """Probability density function at x of the given RV."""
        args, loc, scale = self._parse_args(*args, **kwds)
        x, loc, scale = map(asarray, (x, loc, scale))
        args = tuple(map(asarray, args))
        dtyp = np.promote_types(x.dtype, np.float64)
        x = np.asarray((x - loc) / scale, dtype=dtyp)
        cond0 = self._argcheck(*args) & (scale > 0)
        cond1 = self._support_mask(x, *args) & (scale > 0)
        cond = cond0 & cond1
        output = zeros(shape(cond), dtyp)
        putmask(output, (1 - cond0) + np.isnan(x), self.badvalue)
        if np.any(cond):
            goodargs = argsreduce(cond, *((x,) + args + (scale,)))
            scale, goodargs = goodargs[-1], goodargs[:-1]
            place(output, cond, self._pdf(*goodargs) / scale)
        if output.ndim == 0:
            return output[()]
        return output

    def logpdf(self, x, *args, **kwds):
        """Log of the probability density function at x of the given RV."""
        args, loc, scale = self._parse_args(*args, **kwds)
        x, loc, scale = map(asarray, (x, loc, scale))
        args = tuple(map(asarray, args))
        dtyp = np.promote_types(x.dtype, np.float64)
        x = np.asarray((x - loc) / scale, dtype=dtyp)
        cond0 = self._argcheck(*args) & (scale > 0)
        cond1 = self._support_mask(x, *args) & (scale > 0)
        cond = cond0 & cond1
        output = empty(shape(cond), dtyp)
        output.fill(-inf)
        putmask(output, (1 - cond0) + np.isnan(x), self.badvalue)
        if np.any(cond):
            goodargs = argsreduce(cond, *((x,) + args + (scale,)))
            scale, goodargs = goodargs[-1], goodargs[:-1]
            place(output, cond, self._logpdf(*goodargs) - log(scale))
        if output.ndim == 0:
            return output[()]
        return output

    def cdf(self, x, *args, **kwds):
        """Cumulative distribution function of the given RV."""
        args, loc, scale = self._parse_args(*args, **kwds)
        x, loc, scale = map(asarray, (x, loc, scale))
        args = tuple(map(asarray, args))
        _a, _b = self._get_support(*args)
        dtyp = np.promote_types(x.dtype, np.float64)
        x = np.asarray((x - loc) / scale, dtype=dtyp)
        cond0 = self._argcheck(*args) & (scale > 0)
        cond1 = self._open_support_mask(x, *args) & (scale > 0)
        cond2 = (x >= np.asarray(_b)) & cond0
        cond = cond0 & cond1
        output = zeros(shape(cond), dtyp)
        place(output, (1 - cond0) + np.isnan(x), self.badvalue)
        place(output, cond2, 1.0)
        if np.any(cond):  # call only if at least 1 entry
            goodargs = argsreduce(cond, *((x,) + args))
            place(output, cond, self._cdf(*goodargs))
        if output.ndim == 0:
            return output[()]
        return output

    def logcdf(self, x, *args, **kwds):
        """Log of the cumulative distribution function at x of the given RV."""
        args, loc, scale = self._parse_args(*args, **kwds)
        x, loc, scale = map(asarray, (x, loc, scale))
        args = tuple(map(asarray, args))
        _a, _b = self._get_support(*args)
        dtyp = np.promote_types(x.dtype, np.float64)
        x = np.asarray((x - loc) / scale, dtype=dtyp)
        cond0 = self._argcheck(*args) & (scale > 0)
        cond1 = self._open_support_mask(x, *args) & (scale > 0)
        cond2 = (x >= _b) & cond0
        cond = cond0 & cond1
        output = empty(shape(cond), dtyp)
        output.fill(-inf)
        place(output, (1 - cond0) * (cond1 == cond1) + np.isnan(x), self.badvalue)
        place(output, cond2, 0.0)
        if np.any(cond):  # call only if at least 1 entry
            goodargs = argsreduce(cond, *((x,) + args))
            place(output, cond, self._logcdf(*goodargs))
        if output.ndim == 0:
            return output[()]
        return output

    def sf(self, x, *args, **kwds):
        """Survival function (1 - `cdf`) at x of the given RV."""
        args, loc, scale = self._parse_args(*args, **kwds)
        x, loc, scale = map(asarray, (x, loc, scale))
        args = tuple(map(asarray, args))
        _a, _b = self._get_support(*args)
        dtyp = np.promote_types(x.dtype, np.float64)
        x = np.asarray((x - loc) / scale, dtype=dtyp)
        cond0 = self._argcheck(*args) & (scale > 0)
        cond1 = self._open_support_mask(x, *args) & (scale > 0)
        cond2 = cond0 & (x <= _a)
        cond = cond0 & cond1
        output = zeros(shape(cond), dtyp)
        place(output, (1 - cond0) + np.isnan(x), self.badvalue)
        place(output, cond2, 1.0)
        if np.any(cond):
            goodargs = argsreduce(cond, *((x,) + args))
            place(output, cond, self._sf(*goodargs))
        if output.ndim == 0:
            return output[()]
        return output

    def logsf(self, x, *args, **kwds):
        """Log of the survival function of the given RV."""
        args, loc, scale = self._parse_args(*args, **kwds)
        x, loc, scale = map(asarray, (x, loc, scale))
        args = tuple(map(asarray, args))
        _a, _b = self._get_support(*args)
        dtyp = np.promote_types(x.dtype, np.float64)
        x = np.asarray((x - loc) / scale, dtype=dtyp)
        cond0 = self._argcheck(*args) & (scale > 0)
        cond1 = self._open_support_mask(x, *args) & (scale > 0)
        cond2 = cond0 & (x <= _a)
        cond = cond0 & cond1
        output = empty(shape(cond), dtyp)
        output.fill(-inf)
        place(output, (1 - cond0) + np.isnan(x), self.badvalue)
        place(output, cond2, 0.0)
        if np.any(cond):
            goodargs = argsreduce(cond, *((x,) + args))
            place(output, cond, self._logsf(*goodargs))
        if output.ndim == 0:
            return output[()]
        return output

    def ppf(self, q, *args, **kwds):
        """Percent point function (inverse of `cdf`) at q of the given RV."""
        args, loc, scale = self._parse_args(*args, **kwds)
        q, loc, scale = map(asarray, (q, loc, scale))
        args = tuple(map(asarray, args))
        _a, _b = self._get_support(*args)
        cond0 = self._argcheck(*args) & (scale > 0) & (loc == loc)
        cond1 = (0 < q) & (q < 1)
        cond2 = cond0 & (q == 0)
        cond3 = cond0 & (q == 1)
        cond = cond0 & cond1
        output = np.full(shape(cond), fill_value=self.badvalue)

        lower_bound = _a * scale + loc
        upper_bound = _b * scale + loc
        place(output, cond2, argsreduce(cond2, lower_bound)[0])
        place(output, cond3, argsreduce(cond3, upper_bound)[0])

        if np.any(cond):  # call only if at least 1 entry
            goodargs = argsreduce(cond, *((q,) + args + (scale, loc)))
            scale, loc, goodargs = goodargs[-2], goodargs[-1], goodargs[:-2]
            place(output, cond, self._ppf(*goodargs) * scale + loc)
        if output.ndim == 0:
            return output[()]
        return output

    def isf(self, q, *args, **kwds):
        """Inverse survival function (inverse of `sf`) at q of the given RV."""
        args, loc, scale = self._parse_args(*args, **kwds)
        q, loc, scale = map(asarray, (q, loc, scale))
        args = tuple(map(asarray, args))
        _a, _b = self._get_support(*args)
        cond0 = self._argcheck(*args) & (scale > 0) & (loc == loc)
        cond1 = (0 < q) & (q < 1)
        cond2 = cond0 & (q == 1)
        cond3 = cond0 & (q == 0)
        cond = cond0 & cond1
        output = np.full(shape(cond), fill_value=self.badvalue)

        lower_bound = _a * scale + loc
        upper_bound = _b * scale + loc
        place(output, cond2, argsreduce(cond2, lower_bound)[0])
        place(output, cond3, argsreduce(cond3, upper_bound)[0])

        if np.any(cond):
            goodargs = argsreduce(cond, *((q,) + args + (scale, loc)))
            scale, loc, goodargs = goodargs[-2], goodargs[-1], goodargs[:-2]
            place(output, cond, self._isf(*goodargs) * scale + loc)
        if output.ndim == 0:
            return output[()]
        return output

    def _unpack_loc_scale(self, theta):
        try:
            loc = theta[-2]
            scale = theta[-1]
            args = tuple(theta[:-2])
        except IndexError as e:
            raise ValueError("Not enough input arguments.") from e
        return loc, scale, args

    def _nnlf_and_penalty(self, x, args):
        cond0 = ~self._support_mask(x, *args)
        n_bad = np.count_nonzero(cond0)
        if n_bad > 0:
            x = argsreduce(~cond0, x)[0]
        total, bad_count = _sum_finite(self._logpdf(x, *args))
        n_bad += bad_count
        return -total + n_bad * _LOGXMAX * 100

    def _penalized_nnlf(self, theta, x):
        """Penalized negative loglikelihood function: out-of-support points cost a lot."""
        loc, scale, args = self._unpack_loc_scale(theta)
        if not self._argcheck(*args) or scale <= 0:
            return inf
        x = (x - loc) / scale
        n_log_scale = len(x) * log(scale)
        return self._nnlf_and_penalty(x, args) + n_log_scale

    def _fitstart(self, data, args=None):
        """Starting point for fit (shape arguments + loc + scale)."""
        if args is None:
            args = (1.0,) * self.numargs
        loc, scale = self._fit_loc_scale_support(data, *args)
        return args + (loc, scale)

    def _reduce_func(self, args, kwds, data=None):
        # Convert fixed shape parameters to the standard numeric form: e.g. for
        # stats.beta, shapes='a, b'. To fix `a`, the caller can give a value
        # for `f0`, `fa` or 'fix_a'.  The following converts the latter two
        # into the first (numeric) form.
        shapes = []
        if self.shapes:
            shapes = self.shapes.replace(",", " ").split()
            for j, s in enumerate(shapes):
                key = "f" + str(j)
                names = [key, "f" + s, "fix_" + s]
                val = _get_fixed_fit_value(kwds, names)
                if val is not None:
                    kwds[key] = val

        args = list(args)
        Nargs = len(args)
        fixedn = []
        names = [f"f{n}" for n in range(Nargs - 2)] + ["floc", "fscale"]
        x0 = []
        for n, key in enumerate(names):
            if key in kwds:
                fixedn.append(n)
                args[n] = kwds.pop(key)
            else:
                x0.append(args[n])

        methods = {"mle", "mm"}
        method = kwds.pop("method", "mle").lower()
        if method == "mm":
            n_params = len(shapes) + 2 - len(fixedn)
            exponents = (np.arange(1, n_params + 1))[:, np.newaxis]
            data_moments = np.sum(data[None, :] ** exponents / len(data), axis=1)

            def objective(theta, x):
                return self._moment_error(theta, x, data_moments)

        elif method == "mle":
            objective = self._penalized_nnlf
        else:
            raise ValueError(f"Method '{method}' not available; must be one of {methods}")

        if len(fixedn) == 0:
            func = objective
            restore = None
        else:
            if len(fixedn) == Nargs:
                raise ValueError("All parameters fixed. There is nothing to optimize.")

            def restore(args, theta):
                # Replace with theta for all numbers not in fixedn
                # This allows the non-fixed values to vary, but
                #  we still call self.nnlf with all parameters.
                i = 0
                for n in range(Nargs):
                    if n not in fixedn:
                        args[n] = theta[i]
                        i += 1
                return args

            def func(theta, x):
                newtheta = restore(args[:], theta)
                return objective(newtheta, x)

        return x0, func, restore, args

    def _moment_error(self, theta, x, data_moments):
        loc, scale, args = self._unpack_loc_scale(theta)
        if not self._argcheck(*args) or scale <= 0:
            return inf

        dist_moments = np.array(
            [self.moment(i + 1, *args, loc=loc, scale=scale) for i in range(len(data_moments))]
        )
        if np.any(np.isnan(dist_moments)):
            raise ValueError(
                "Method of moments encountered a non-finite "
                "distribution moment and cannot continue. "
                "Consider trying method='MLE'."
            )

        return (((data_moments - dist_moments) / np.maximum(np.abs(data_moments), 1e-8)) ** 2).sum()

    def fit(self, data, *args, **kwds):
        """Estimate shape, location and scale parameters from data by MLE or MM.

        Fitting minimizes with ``scipy.optimize``.
        """
        from scipy import optimize
        from scipy.stats._warnings_errors import FitError

        method = kwds.get("method", "mle").lower()

        Narg = len(args)
        if Narg > self.numargs:
            raise TypeError("Too many input arguments.")

        # Note: `ravel()` is called for backwards compatibility.
        data = np.asarray(data).ravel()
        if not np.isfinite(data).all():
            raise ValueError("The data contains non-finite values.")

        start = [None] * 2
        if (Narg < self.numargs) or not ("loc" in kwds and "scale" in kwds):
            # get distribution specific starting locations
            start = self._fitstart(data)
            args += start[Narg:-2]
        loc = kwds.pop("loc", start[-2])
        scale = kwds.pop("scale", start[-1])
        args += (loc, scale)
        x0, func, restore, args = self._reduce_func(args, kwds, data=data)
        optimizer = kwds.pop("optimizer", optimize.fmin)
        # convert string to function in scipy.optimize
        optimizer = _fit_determine_optimizer(optimizer)
        # by now kwds must be empty, since everybody took what they needed
        if kwds:
            raise TypeError(f"Unknown arguments: {kwds}.")

        # Minimizing the sum of squared errors also covers method of moments cases
        # with no exact solution.
        vals = optimizer(func, x0, args=(data,), disp=0)
        obj = func(vals, data)

        if restore is not None:
            vals = restore(args, vals)
        vals = tuple(vals)

        loc, scale, shapes = self._unpack_loc_scale(vals)
        if not (np.all(self._argcheck(*shapes)) and scale > 0):
            raise FitError(
                "Optimization converged to parameters that are "
                "outside the range allowed by the distribution."
            )

        if method == "mm":
            if not np.isfinite(obj):
                raise FitError(
                    "Optimization failed: either a data moment "
                    "or fitted distribution moment is "
                    "non-finite."
                )

        return vals

    def _fit_loc_scale_support(self, data, *args):
        """Estimate loc and scale parameters from data accounting for support."""
        data = np.asarray(data)

        # Estimate location and scale according to the method of moments.
        loc_hat, scale_hat = self.fit_loc_scale(data, *args)

        # Compute the support according to the shape parameters.
        self._argcheck(*args)
        _a, _b = self._get_support(*args)
        a, b = _a, _b
        support_width = b - a

        # If the support is empty then return the moment-based estimates.
        if support_width <= 0:
            return loc_hat, scale_hat

        # Compute the proposed support according to the loc and scale
        # estimates.
        a_hat = loc_hat + a * scale_hat
        b_hat = loc_hat + b * scale_hat

        # Use the moment-based estimates if they are compatible with the data.
        data_a = np.min(data)
        data_b = np.max(data)
        if a_hat < data_a and data_b < b_hat:
            return loc_hat, scale_hat

        # Otherwise find other estimates that are compatible with the data.
        data_width = data_b - data_a
        rel_margin = 0.1
        margin = data_width * rel_margin

        # For a finite interval, both the location and scale
        # should have interesting values.
        if support_width < np.inf:
            loc_hat = (data_a - a) - margin
            scale_hat = (data_width + 2 * margin) / support_width
            return loc_hat, scale_hat

        # For a one-sided interval, use only an interesting location parameter.
        if a > -np.inf:
            return (data_a - a) - margin, 1
        elif b < np.inf:
            return (data_b - b) + margin, 1
        else:
            raise RuntimeError

    def fit_loc_scale(self, data, *args):
        """Estimate loc and scale parameters from data using 1st and 2nd moments."""
        mu, mu2 = self.stats(*args, **{"moments": "mv"})
        tmp = asarray(data)
        muhat = tmp.mean()
        mu2hat = tmp.var()
        Shat = sqrt(mu2hat / mu2)
        with np.errstate(invalid="ignore"):
            Lhat = muhat - Shat * mu
        if not np.isfinite(Lhat):
            Lhat = 0
        if not (np.isfinite(Shat) and (0 < Shat)):
            Shat = 1
        return Lhat, Shat

    def _entropy(self, *args):
        from scipy import integrate

        def integ(x):
            val = self._pdf(x, *args)
            return entr(val)

        # upper limit is often inf, so suppress warnings when integrating
        _a, _b = self._get_support(*args)
        with np.errstate(over="ignore"):
            h = integrate.quad(integ, _a, _b)[0]

        if not np.isnan(h):
            return h
        else:
            # try with different limits if integration problems
            low, upp = self.ppf([1e-10, 1.0 - 1e-10], *args)
            if np.isinf(_b):
                upper = upp
            else:
                upper = _b
            if np.isinf(_a):
                lower = low
            else:
                lower = _a
            return integrate.quad(integ, lower, upper)[0]

    def expect(
        self, func=None, args=(), loc=0, scale=1, lb=None, ub=None, conditional=False, **kwds
    ):
        """Expected value of a function with respect to the distribution, by integration."""
        from scipy import integrate

        lockwds = {"loc": loc, "scale": scale}
        self._argcheck(*args)
        _a, _b = self._get_support(*args)
        if func is None:

            def fun(x, *args):
                return x * self.pdf(x, *args, **lockwds)

        else:

            def fun(x, *args):
                return func(x) * self.pdf(x, *args, **lockwds)

        if lb is None:
            lb = loc + _a * scale
        if ub is None:
            ub = loc + _b * scale

        cdf_bounds = self.cdf([lb, ub], *args, **lockwds)
        invfac = cdf_bounds[1] - cdf_bounds[0]

        kwds["args"] = args

        # split interval to help integrator w/ infinite support; see gh-8928
        alpha = 0.05  # split body from tails at probability mass `alpha`
        inner_bounds = np.array([alpha, 1 - alpha])
        cdf_inner_bounds = cdf_bounds[0] + invfac * inner_bounds
        c, d = loc + self._ppf(cdf_inner_bounds, *args) * scale

        # Do not silence warnings from integration.
        lbc = integrate.quad(fun, lb, c, **kwds)[0]
        cd = integrate.quad(fun, c, d, **kwds)[0]
        dub = integrate.quad(fun, d, ub, **kwds)[0]
        vals = lbc + cd + dub

        if conditional:
            vals /= invfac
        return np.array(vals)[()]  # make it a numpy scalar like other methods

    def _delta_cdf(self, x1, x2, *args, loc=0, scale=1):
        """CDF(x2) - CDF(x1), computed from the survival function above the median."""
        cdf1 = self.cdf(x1, *args, loc=loc, scale=scale)
        result = np.where(
            cdf1 > 0.5,
            (self.sf(x1, *args, loc=loc, scale=scale) - self.sf(x2, *args, loc=loc, scale=scale)),
            self.cdf(x2, *args, loc=loc, scale=scale) - cdf1,
        )
        if result.ndim == 0:
            result = result[()]
        return result


def _get_fixed_fit_value(kwds, names):
    """The value of the one fixed-parameter keyword in ``names``, removed from ``kwds``."""
    vals = [(name, kwds.pop(name)) for name in names if name in kwds]
    if len(vals) > 1:
        repeated = [name for name, val in vals]
        raise ValueError(
            "fit method got multiple keyword arguments to "
            "specify the same fixed parameter: " + ", ".join(repeated)
        )
    return vals[0][1] if vals else None


def _fit_determine_optimizer(optimizer):
    from scipy import optimize

    if not callable(optimizer) and isinstance(optimizer, str):
        if not optimizer.startswith("fmin_"):
            optimizer = "fmin_" + optimizer
        if optimizer == "fmin_":
            optimizer = "fmin"
        try:
            optimizer = getattr(optimize, optimizer)
        except AttributeError as e:
            raise ValueError(f"{optimizer} is not a valid optimizer") from e
    return optimizer


# Helpers for the discrete distributions
def _drv2_moment(self, n, *args):
    """Non-central moment of discrete distribution."""

    def fun(x):
        return np.power(x, n) * self._pmf(x, *args)

    _a, _b = self._get_support(*args)
    return _expect(fun, _a, _b, self._ppf(0.5, *args), self.inc)


def _drv2_ppfsingle(self, q, *args):  # Use basic bisection algorithm
    _a, _b = self._get_support(*args)
    b = _b
    a = _a

    step = 10
    if isinf(b):  # Be sure ending point is > q
        b = float(max(100 * q, 10))
        while 1:
            if b >= _b:
                qb = 1.0
                break
            qb = self._cdf(b, *args)
            if qb < q:
                b += step
                step *= 2
            else:
                break
    else:
        qb = 1.0

    step = 10
    if isinf(a):  # be sure starting point < q
        a = float(min(-100 * q, -10))
        while 1:
            if a <= _a:
                qb = 0.0
                break
            qa = self._cdf(a, *args)
            if qa > q:
                a -= step
                step *= 2
            else:
                break
    else:
        qa = self._cdf(a, *args)

    if np.isinf(a) or np.isinf(b):
        message = "Arguments that bracket the requested quantile could not be found."
        raise RuntimeError(message)

    # maximum number of bisections within the normal float64s
    # maxiter = int(np.log2(finfo.max) - np.log2(finfo.smallest_normal))
    maxiter = 2046
    for i in range(maxiter):
        if qa == q:
            return a
        if qb == q:
            return b
        if b <= a + 1:
            if qa > q:
                return a
            else:
                return b
        c = int((a + b) / 2.0)
        qc = self._cdf(c, *args)
        if qc < q:
            if a != c:
                a = c
            else:
                raise RuntimeError("updating stopped, endless loop")
            qa = qc
        elif qc > q:
            if b != c:
                b = c
            else:
                raise RuntimeError("updating stopped, endless loop")
            qb = qc
        else:
            return c


# Must over-ride one of _pmf or _cdf or pass in
#  x_k, p(x_k) lists in initialization


class rv_discrete(rv_generic):
    """A generic discrete random variable class meant for subclassing."""

    def __init__(
        self,
        a=0,
        b=inf,
        name=None,
        badvalue=None,
        moment_tol=1e-8,
        values=None,
        inc=1,
        longname=None,
        shapes=None,
        seed=None,
    ):
        if values is not None:
            raise NotImplementedError(
                "rv_discrete(values=...) is not supported by shellsim's SciPy"
            )

        super().__init__(seed)

        # cf generic freeze
        self._ctor_param = dict(
            a=a,
            b=b,
            name=name,
            badvalue=badvalue,
            moment_tol=moment_tol,
            values=values,
            inc=inc,
            longname=longname,
            shapes=shapes,
            seed=seed,
        )

        if badvalue is None:
            badvalue = nan
        self.badvalue = badvalue
        self.a = a
        self.b = b
        self.moment_tol = moment_tol
        self.inc = inc
        self.shapes = shapes

        # scale=1 for discrete RVs
        self._construct_argparser(
            meths_to_inspect=[self._pmf, self._cdf], locscale=(("loc", 0),)
        )
        self._attach_methods()
        if name is None:
            name = "Distribution"
        self.name = name

    def _attach_methods(self):
        self._cdfvec = vectorize(self._cdf_single, otypes="d")
        self.vecentropy = vectorize(self._entropy)

        # SciPy binds these vectorized helpers with ``types.MethodType``; ``self`` is then
        # their first, broadcast, argument.
        _vec_generic_moment = vectorize(_drv2_moment, otypes="d")
        _vec_generic_moment.nin = self.numargs + 2
        self.generic_moment = lambda *args: _vec_generic_moment(self, *args)

        _vppf = vectorize(_drv2_ppfsingle, otypes="d")
        _vppf.nin = self.numargs + 2
        self._ppfvec = lambda *args: _vppf(self, *args)

        self._cdfvec.nin = self.numargs + 1

    def _updated_ctor_param(self):
        dct = self._ctor_param.copy()
        dct["a"] = self.a
        dct["b"] = self.b
        dct["badvalue"] = self.badvalue
        dct["moment_tol"] = self.moment_tol
        dct["inc"] = self.inc
        dct["name"] = self.name
        dct["shapes"] = self.shapes
        return dct

    def _nonzero(self, k, *args):
        return floor(k) == k

    def _pmf(self, k, *args):
        return self._cdf(k, *args) - self._cdf(k - 1, *args)

    def _logpmf(self, k, *args):
        with np.errstate(divide="ignore"):
            return log(self._pmf(k, *args))

    def _logpxf(self, k, *args):
        # continuous distributions have PDF, discrete have PMF, but sometimes
        # the distinction doesn't matter.
        return self._logpmf(k, *args)

    def _unpack_loc_scale(self, theta):
        try:
            loc = theta[-1]
            scale = 1
            args = tuple(theta[:-1])
        except IndexError as e:
            raise ValueError("Not enough input arguments.") from e
        return loc, scale, args

    def _cdf_single(self, k, *args):
        _a, _b = self._get_support(*args)
        m = arange(int(_a), k + 1)
        return np.sum(self._pmf(m, *args), axis=0)

    def _cdf(self, x, *args):
        k = floor(x).astype(np.float64)
        return self._cdfvec(k, *args)

    # generic _logcdf, _sf, _logsf, _ppf, _isf, _rvs defined in rv_generic

    def rvs(self, *args, **kwargs):
        """Random variates of given type."""
        kwargs["discrete"] = True
        return super().rvs(*args, **kwargs)

    def pmf(self, k, *args, **kwds):
        """Probability mass function at k of the given RV."""
        args, loc, _ = self._parse_args(*args, **kwds)
        k, loc = map(asarray, (k, loc))
        args = tuple(map(asarray, args))
        _a, _b = self._get_support(*args)
        k = asarray(k - loc)
        cond0 = self._argcheck(*args)
        cond1 = (k >= _a) & (k <= _b)
        cond1 = cond1 & self._nonzero(k, *args)
        cond = cond0 & cond1
        output = zeros(shape(cond), "d")
        place(output, (1 - cond0) + np.isnan(k), self.badvalue)
        if np.any(cond):
            goodargs = argsreduce(cond, *((k,) + args))
            place(output, cond, np.clip(self._pmf(*goodargs), 0, 1))
        if output.ndim == 0:
            return output[()]
        return output

    def logpmf(self, k, *args, **kwds):
        """Log of the probability mass function at k of the given RV."""
        args, loc, _ = self._parse_args(*args, **kwds)
        k, loc = map(asarray, (k, loc))
        args = tuple(map(asarray, args))
        _a, _b = self._get_support(*args)
        k = asarray(k - loc)
        cond0 = self._argcheck(*args)
        cond1 = (k >= _a) & (k <= _b)
        cond1 = cond1 & self._nonzero(k, *args)
        cond = cond0 & cond1
        output = empty(shape(cond), "d")
        output.fill(-inf)
        place(output, (1 - cond0) + np.isnan(k), self.badvalue)
        if np.any(cond):
            goodargs = argsreduce(cond, *((k,) + args))
            place(output, cond, self._logpmf(*goodargs))
        if output.ndim == 0:
            return output[()]
        return output

    def cdf(self, k, *args, **kwds):
        """Cumulative distribution function of the given RV."""
        args, loc, _ = self._parse_args(*args, **kwds)
        k, loc = map(asarray, (k, loc))
        args = tuple(map(asarray, args))
        _a, _b = self._get_support(*args)
        k = asarray(k - loc)
        cond0 = self._argcheck(*args)
        cond1 = (k >= _a) & (k < _b)
        cond2 = k >= _b
        cond3 = np.isneginf(k)
        cond = cond0 & cond1 & np.isfinite(k)

        output = zeros(shape(cond), "d")
        place(output, cond2 * (cond0 == cond0), 1.0)
        place(output, cond3 * (cond0 == cond0), 0.0)
        place(output, (1 - cond0) + np.isnan(k), self.badvalue)

        if np.any(cond):
            goodargs = argsreduce(cond, *((k,) + args))
            place(output, cond, np.clip(self._cdf(*goodargs), 0, 1))
        if output.ndim == 0:
            return output[()]
        return output

    def logcdf(self, k, *args, **kwds):
        """Log of the cumulative distribution function at k of the given RV."""
        args, loc, _ = self._parse_args(*args, **kwds)
        k, loc = map(asarray, (k, loc))
        args = tuple(map(asarray, args))
        _a, _b = self._get_support(*args)
        k = asarray(k - loc)
        cond0 = self._argcheck(*args)
        cond1 = (k >= _a) & (k < _b)
        cond2 = k >= _b
        cond = cond0 & cond1
        output = empty(shape(cond), "d")
        output.fill(-inf)
        place(output, (1 - cond0) + np.isnan(k), self.badvalue)
        place(output, cond2 * (cond0 == cond0), 0.0)

        if np.any(cond):
            goodargs = argsreduce(cond, *((k,) + args))
            place(output, cond, self._logcdf(*goodargs))
        if output.ndim == 0:
            return output[()]
        return output

    def sf(self, k, *args, **kwds):
        """Survival function (1 - `cdf`) at k of the given RV."""
        args, loc, _ = self._parse_args(*args, **kwds)
        k, loc = map(asarray, (k, loc))
        args = tuple(map(asarray, args))
        _a, _b = self._get_support(*args)
        k = asarray(k - loc)
        cond0 = self._argcheck(*args)
        cond1 = (k >= _a) & (k < _b)
        cond2 = ((k < _a) | np.isneginf(k)) & cond0
        cond = cond0 & cond1 & np.isfinite(k)
        output = zeros(shape(cond), "d")
        place(output, (1 - cond0) + np.isnan(k), self.badvalue)
        place(output, cond2, 1.0)
        if np.any(cond):
            goodargs = argsreduce(cond, *((k,) + args))
            place(output, cond, np.clip(self._sf(*goodargs), 0, 1))
        if output.ndim == 0:
            return output[()]
        return output

    def logsf(self, k, *args, **kwds):
        """Log of the survival function of the given RV."""
        args, loc, _ = self._parse_args(*args, **kwds)
        k, loc = map(asarray, (k, loc))
        args = tuple(map(asarray, args))
        _a, _b = self._get_support(*args)
        k = asarray(k - loc)
        cond0 = self._argcheck(*args)
        cond1 = (k >= _a) & (k < _b)
        cond2 = (k < _a) & cond0
        cond = cond0 & cond1
        output = empty(shape(cond), "d")
        output.fill(-inf)
        place(output, (1 - cond0) + np.isnan(k), self.badvalue)
        place(output, cond2, 0.0)
        if np.any(cond):
            goodargs = argsreduce(cond, *((k,) + args))
            place(output, cond, self._logsf(*goodargs))
        if output.ndim == 0:
            return output[()]
        return output

    def ppf(self, q, *args, **kwds):
        """Percent point function (inverse of `cdf`) at q of the given RV."""
        args, loc, _ = self._parse_args(*args, **kwds)
        q, loc = map(asarray, (q, loc))
        args = tuple(map(asarray, args))
        _a, _b = self._get_support(*args)
        cond0 = self._argcheck(*args) & (loc == loc)
        cond1 = (q > 0) & (q < 1)
        cond2 = (q == 0) & cond0
        cond3 = (q == 1) & cond0
        cond = cond0 & cond1
        # output type 'd' to handle nin and inf
        output = np.full(shape(cond), fill_value=self.badvalue, dtype="d")

        place(output, cond2, argsreduce(cond2, _a - 1 + loc)[0])
        place(output, cond3, argsreduce(cond3, _b + loc)[0])
        if np.any(cond):
            goodargs = argsreduce(cond, *((q,) + args + (loc,)))
            loc, goodargs = goodargs[-1], goodargs[:-1]
            place(output, cond, self._ppf(*goodargs) + loc)

        if output.ndim == 0:
            return output[()]
        return output

    def isf(self, q, *args, **kwds):
        """Inverse survival function (inverse of `sf`) at q of the given RV."""
        args, loc, _ = self._parse_args(*args, **kwds)
        q, loc = map(asarray, (q, loc))
        args = tuple(map(asarray, args))
        _a, _b = self._get_support(*args)
        cond0 = self._argcheck(*args) & (loc == loc)
        cond1 = (q > 0) & (q < 1)
        cond2 = (q == 1) & cond0
        cond3 = (q == 0) & cond0
        cond = cond0 & cond1

        # output type 'd' to handle nin and inf
        output = np.full(shape(cond), fill_value=self.badvalue, dtype="d")
        lower_bound = _a - 1 + loc
        upper_bound = _b + loc
        place(output, cond2, argsreduce(cond2, lower_bound)[0])
        place(output, cond3, argsreduce(cond3, upper_bound)[0])

        # call place only if at least 1 valid argument
        if np.any(cond):
            goodargs = argsreduce(cond, *((q,) + args + (loc,)))
            loc, goodargs = goodargs[-1], goodargs[:-1]
            place(output, cond, self._isf(*goodargs) + loc)

        if output.ndim == 0:
            return output[()]
        return output

    def _entropy(self, *args):
        _a, _b = self._get_support(*args)
        return _expect(
            lambda x: entr(self._pmf(x, *args)), _a, _b, self._ppf(0.5, *args), self.inc
        )

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
        """Expected value of a function with respect to the distribution, by summation."""
        args, _, _ = self._parse_args(*args)

        if func is None:

            def fun(x):
                # loc and args from outer scope
                return (x + loc) * self._pmf(x, *args)

        else:

            def fun(x):
                # loc and args from outer scope
                return func(x + loc) * self._pmf(x, *args)

        _a, _b = self._get_support(*args)
        if lb is None:
            lb = _a
        else:
            lb = lb - loc  # convert bound for standardized distribution
        if ub is None:
            ub = _b
        else:
            ub = ub - loc  # convert bound for standardized distribution
        if conditional:
            invfac = self.sf(lb - 1, *args) - self.sf(ub, *args)
        else:
            invfac = 1.0

        # iterate over the support, starting from the median
        x0 = self._ppf(0.5, *args)
        res = _expect(fun, lb, ub, x0, self.inc, maxcount, tolerance, chunksize)
        return res / invfac


def _expect(fun, lb, ub, x0, inc, maxcount=1000, tolerance=1e-10, chunksize=32):
    """Helper for computing the expectation value of `fun`."""
    # short-circuit if the support size is small enough
    if (ub - lb) <= chunksize:
        supp = np.arange(lb, ub + 1, inc)
        vals = fun(supp)
        return np.sum(vals)

    # otherwise, iterate starting from x0
    if x0 < lb:
        x0 = lb
    if x0 > ub:
        x0 = ub

    count, tot = 0, 0.0
    # iterate over [x0, ub] inclusive
    for x in _iter_chunked(x0, ub + 1, chunksize=chunksize, inc=inc):
        count += x.size
        delta = np.sum(fun(x))
        tot += delta
        if abs(delta) < tolerance * x.size:
            break
        if count > maxcount:
            warnings.warn("expect(): sum did not converge", RuntimeWarning, stacklevel=3)
            return tot

    # iterate over [lb, x0)
    for x in _iter_chunked(x0 - 1, lb - 1, chunksize=chunksize, inc=-inc):
        count += x.size
        delta = np.sum(fun(x))
        tot += delta
        if abs(delta) < tolerance * x.size:
            break
        if count > maxcount:
            warnings.warn("expect(): sum did not converge", RuntimeWarning, stacklevel=3)
            break

    return tot


def _iter_chunked(x0, x1, chunksize=4, inc=1):
    """Iterate from x0 to x1 in chunks of chunksize and steps inc.

    x0 must be finite, x1 need not be; in the latter case the iterator is infinite. Iterates
    downwards when x0 > x1, which needs a negative inc.
    """
    if inc == 0:
        raise ValueError("Cannot increment by zero.")
    if chunksize <= 0:
        raise ValueError(f"Chunk size must be positive; got {chunksize}.")

    s = 1 if inc > 0 else -1
    stepsize = abs(chunksize * inc)

    x = np.copy(x0)
    while (x - x1) * inc < 0:
        delta = min(stepsize, abs(x - x1))
        step = delta * s
        supp = np.arange(x, x + step, inc)
        x += step
        yield supp
