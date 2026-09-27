"""Continuous distributions, following SciPy 1.18's ``scipy/stats/_continuous_distns.py``.

shellsim implements ``norm``, ``t``, ``chi2``, ``f``, ``uniform`` and ``expon``. Their methods
are SciPy's formulas over the ``scipy.special`` ufuncs. ``norm``, ``uniform`` and ``expon`` fit
by SciPy's closed-form maximum-likelihood estimates; the method of moments and other fits go
through the generic ``rv_continuous.fit``, which needs ``scipy.optimize``.
"""

import numpy as np
import scipy.special as sc
from scipy._lib._util import _lazyselect, apply_where
from scipy.stats._distn_infrastructure import rv_continuous


def _remove_optimizer_parameters(kwds):
    """Remove 'loc', 'scale', 'optimizer' and 'method' from ``kwds``, which must then be empty.

    Used by the ``fit`` methods that do not optimize.
    """
    kwds.pop("loc", None)
    kwds.pop("scale", None)
    kwds.pop("optimizer", None)
    kwds.pop("method", None)
    if kwds:
        raise TypeError(f"Unknown arguments: {kwds}.")


def _fit_method_is_mm(kwds):
    # SciPy's ``_call_super_mom`` decorator sends the method of moments to the generic fit.
    return kwds.get("method", "mle").lower() == "mm"


# loc = mu, scale = std
# Keep these implementations out of the class definition so they can be reused
# by other distributions.
_norm_pdf_C = np.sqrt(2 * np.pi)
_norm_pdf_logC = np.log(_norm_pdf_C)


def _norm_pdf(x):
    return np.exp(-(x**2) / 2.0) / _norm_pdf_C


def _norm_logpdf(x):
    return -(x**2) / 2.0 - _norm_pdf_logC


def _norm_cdf(x):
    return sc.ndtr(x)


def _norm_logcdf(x):
    return sc.log_ndtr(x)


def _norm_ppf(q):
    return sc.ndtri(q)


def _norm_sf(x):
    return _norm_cdf(-x)


def _norm_logsf(x):
    return _norm_logcdf(-x)


def _norm_isf(q):
    return -_norm_ppf(q)


class norm_gen(rv_continuous):
    """A normal continuous random variable.

    The location (``loc``) keyword specifies the mean. The scale (``scale``) keyword specifies
    the standard deviation.
    """

    def _rvs(self, size=None, random_state=None):
        return random_state.standard_normal(size)

    def _pdf(self, x):
        # norm.pdf(x) = exp(-x**2/2)/sqrt(2*pi)
        return _norm_pdf(x)

    def _logpdf(self, x):
        return _norm_logpdf(x)

    def _cdf(self, x):
        return _norm_cdf(x)

    def _logcdf(self, x):
        return _norm_logcdf(x)

    def _sf(self, x):
        return _norm_sf(x)

    def _logsf(self, x):
        return _norm_logsf(x)

    def _ppf(self, q):
        return _norm_ppf(q)

    def _isf(self, q):
        return _norm_isf(q)

    def _stats(self):
        return 0.0, 1.0, 0.0, 0.0

    def _entropy(self):
        return 0.5 * (np.log(2 * np.pi) + 1)

    def fit(self, data, *args, **kwds):
        """Maximum likelihood estimates of ``loc`` and ``scale``: the mean and the RMS deviation.

        ``floc`` and ``fscale`` fix a parameter. ``method='MM'`` uses the generic fit.
        """
        if _fit_method_is_mm(kwds):
            return super().fit(data, *args, **kwds)
        if args:
            raise TypeError(
                f"norm_gen.fit() takes 2 positional arguments but {len(args) + 2} were given"
            )
        floc = kwds.pop("floc", None)
        fscale = kwds.pop("fscale", None)

        _remove_optimizer_parameters(kwds)

        if floc is not None and fscale is not None:
            # This check is for consistency with `rv_continuous.fit`.
            # Without this check, this function would just return the
            # parameters that were given.
            raise ValueError("All parameters fixed. There is nothing to optimize.")

        data = np.asarray(data)

        if not np.isfinite(data).all():
            raise ValueError("The data contains non-finite values.")

        if floc is None:
            loc = data.mean()
        else:
            loc = floc

        if fscale is None:
            scale = np.sqrt(((data - loc) ** 2).mean())
        else:
            scale = fscale

        return loc, scale

    def _munp(self, n):
        """``n``-th moment of the standard normal: ``(n - 1)!!`` for even ``n``, else 0."""
        if n == 0:
            return 1.0
        if n % 2 == 0:
            return sc.factorial2(int(n) - 1)
        else:
            return 0.0


norm = norm_gen(name="norm")


class FitDataError(ValueError):
    """Raised when input data is inconsistent with fixed parameters."""

    # This exception is raised by, for example, expon_gen.fit when floc is fixed and there
    # are values in the data below it.
    def __init__(self, distr, lower, upper):
        self.args = (
            "Invalid values in `data`.  Maximum likelihood "
            f"estimation with {distr!r} requires that {lower!r} < "
            f"(x - loc)/scale  < {upper!r} for each x in `data`.",
        )


class chi2_gen(rv_continuous):
    """A chi-squared continuous random variable, with ``df`` degrees of freedom."""

    def _rvs(self, df, size=None, random_state=None):
        return random_state.chisquare(df, size)

    def _pdf(self, x, df):
        # chi2.pdf(x, df) = 1 / (2*gamma(df/2)) * (x/2)**(df/2-1) * exp(-x/2)
        return np.exp(self._logpdf(x, df))

    def _logpdf(self, x, df):
        return sc.xlogy(df / 2.0 - 1, x) - x / 2.0 - sc.gammaln(df / 2.0) - (np.log(2) * df) / 2.0

    def _cdf(self, x, df):
        return sc.chdtr(df, x)

    def _sf(self, x, df):
        return sc.chdtrc(df, x)

    def _isf(self, p, df):
        return sc.chdtri(df, p)

    def _ppf(self, p, df):
        return 2 * sc.gammaincinv(df / 2, p)

    def _stats(self, df):
        mu = df
        mu2 = 2 * df
        g1 = 2 * np.sqrt(2.0 / df)
        g2 = 12.0 / df
        return mu, mu2, g1, g2

    def _entropy(self, df):
        half_df = 0.5 * df

        def regular_formula(half_df):
            return (
                half_df + np.log(2) + sc.gammaln(half_df) + (1 - half_df) * sc.psi(half_df)
            )

        def asymptotic_formula(half_df):
            # plug in the above formula the following asymptotic
            # expansions:
            # ln(gamma(a)) ~ (a - 0.5) * ln(a) - a + 0.5 * ln(2 * pi) +
            #                 1/(12 * a) - 1/(360 * a**3)
            # psi(a) ~ ln(a) - 1/(2 * a) - 1/(3 * a**2) + 1/120 * a**4)
            c = np.log(2) + 0.5 * (1 + np.log(2 * np.pi))
            h = 0.5 / half_df
            return h * (-2 / 3 + h * (-1 / 3 + h * (-4 / 45 + h / 7.5))) + 0.5 * np.log(half_df) + c

        return apply_where(half_df < 125, half_df, regular_formula, asymptotic_formula)


chi2 = chi2_gen(a=0.0, name="chi2")


class expon_gen(rv_continuous):
    """An exponential continuous random variable, with ``pdf = exp(-x)`` for ``x >= 0``.

    ``scale = 1 / lambda`` gives the rate parameterization.
    """

    def _rvs(self, size=None, random_state=None):
        return random_state.standard_exponential(size)

    def _pdf(self, x):
        # expon.pdf(x) = exp(-x)
        return np.exp(-x)

    def _logpdf(self, x):
        return -x

    def _cdf(self, x):
        return -sc.expm1(-x)

    def _ppf(self, q):
        return -sc.log1p(-q)

    def _sf(self, x):
        return np.exp(-x)

    def _logsf(self, x):
        return -x

    def _isf(self, q):
        return -np.log(q)

    def _stats(self):
        return 1.0, 1.0, 2.0, 6.0

    def _entropy(self):
        return 1.0

    def fit(self, data, *args, **kwds):
        """Maximum likelihood estimates: ``loc`` is the minimum, ``scale`` the shifted mean.

        ``floc`` and ``fscale`` fix a parameter. ``method='MM'`` uses the generic fit.
        """
        if _fit_method_is_mm(kwds):
            return super().fit(data, *args, **kwds)
        if len(args) > 0:
            raise TypeError("Too many arguments.")

        floc = kwds.pop("floc", None)
        fscale = kwds.pop("fscale", None)

        _remove_optimizer_parameters(kwds)

        if floc is not None and fscale is not None:
            # This check is for consistency with `rv_continuous.fit`.
            raise ValueError("All parameters fixed. There is nothing to optimize.")

        data = np.asarray(data)

        if not np.isfinite(data).all():
            raise ValueError("The data contains non-finite values.")

        data_min = data.min()

        if floc is None:
            # ML estimate of the location is the minimum of the data.
            loc = data_min
        else:
            loc = floc
            if data_min < loc:
                # There are values that are less than the specified loc.
                raise FitDataError("expon", lower=floc, upper=np.inf)

        if fscale is None:
            # ML estimate of the scale is the shifted mean.
            scale = data.mean() - loc
        else:
            scale = fscale

        # We expect the return values to be floating point, so ensure it
        # by explicitly converting to float.
        return float(loc), float(scale)


expon = expon_gen(a=0.0, name="expon")


class f_gen(rv_continuous):
    """An F continuous random variable, with ``dfn`` and ``dfd`` degrees of freedom."""

    def _rvs(self, dfn, dfd, size=None, random_state=None):
        return random_state.f(dfn, dfd, size)

    def _pdf(self, x, dfn, dfd):
        #                      df2**(df2/2) * df1**(df1/2) * x**(df1/2-1)
        # F.pdf(x, df1, df2) = --------------------------------------------
        #                      (df2+df1*x)**((df1+df2)/2) * B(df1/2, df2/2)
        return np.exp(self._logpdf(x, dfn, dfd))

    def _logpdf(self, x, dfn, dfd):
        n = 1.0 * dfn
        m = 1.0 * dfd
        lPx = (
            m / 2 * np.log(m)
            + n / 2 * np.log(n)
            + sc.xlogy(n / 2 - 1, x)
            - (((n + m) / 2) * np.log(m + n * x) + sc.betaln(n / 2, m / 2))
        )
        return lPx

    def _cdf(self, x, dfn, dfd):
        return sc.fdtr(dfn, dfd, x)

    def _sf(self, x, dfn, dfd):
        return sc.fdtrc(dfn, dfd, x)

    def _ppf(self, q, dfn, dfd):
        return sc.fdtri(dfn, dfd, q)

    def _stats(self, dfn, dfd):
        v1, v2 = 1.0 * dfn, 1.0 * dfd
        v2_2, v2_4, v2_6, v2_8 = v2 - 2.0, v2 - 4.0, v2 - 6.0, v2 - 8.0

        mu = apply_where(v2 > 2, (v2, v2_2), lambda v2, v2_2: v2 / v2_2, fill_value=np.inf)

        mu2 = apply_where(
            v2 > 4,
            (v1, v2, v2_2, v2_4),
            lambda v1, v2, v2_2, v2_4: 2 * v2 * v2 * (v1 + v2_2) / (v1 * v2_2**2 * v2_4),
            fill_value=np.inf,
        )

        g1 = apply_where(
            v2 > 6,
            (v1, v2_2, v2_4, v2_6),
            lambda v1, v2_2, v2_4, v2_6: (
                (2 * v1 + v2_2) / v2_6 * np.sqrt(v2_4 / (v1 * (v1 + v2_2)))
            ),
            fill_value=np.nan,
        )
        g1 *= np.sqrt(8.0)

        g2 = apply_where(
            v2 > 8,
            (g1, v2_6, v2_8),
            lambda g1, v2_6, v2_8: (8 + g1 * g1 * v2_6) / v2_8,
            fill_value=np.nan,
        )
        g2 *= 3.0 / 2.0

        return mu, mu2, g1, g2

    def _entropy(self, dfn, dfd):
        # the formula found in literature is incorrect. This one yields the
        # same result as numerical integration using the generic entropy
        # definition.
        half_dfn = 0.5 * dfn
        half_dfd = 0.5 * dfd
        half_sum = 0.5 * (dfn + dfd)

        return (
            np.log(dfd)
            - np.log(dfn)
            + sc.betaln(half_dfn, half_dfd)
            + (1 - half_dfn) * sc.psi(half_dfn)
            - (1 + half_dfd) * sc.psi(half_dfd)
            + half_sum * sc.psi(half_sum)
        )


f = f_gen(a=0.0, name="f")


class t_gen(rv_continuous):
    """A Student's t continuous random variable, with ``df`` degrees of freedom."""

    def _rvs(self, df, size=None, random_state=None):
        return random_state.standard_t(df, size=size)

    def _pdf(self, x, df):
        return apply_where(
            df == np.inf,
            (x, df),
            lambda x, df: norm._pdf(x),
            lambda x, df: np.exp(self._logpdf(x, df)),
        )

    def _logpdf(self, x, df):
        def t_logpdf(x, df):
            return (
                np.log(sc.poch(0.5 * df, 0.5))
                - 0.5 * (np.log(df) + np.log(np.pi))
                - (df + 1) / 2 * np.log1p(x * x / df)
            )

        def norm_logpdf(x, df):
            return norm._logpdf(x)

        return apply_where(df == np.inf, (x, df), norm_logpdf, t_logpdf)

    def _cdf(self, x, df):
        return sc.stdtr(df, x)

    def _sf(self, x, df):
        return sc.stdtr(df, -x)

    def _ppf(self, q, df):
        return sc.stdtrit(df, q)

    def _isf(self, q, df):
        return -sc.stdtrit(df, q)

    def _stats(self, df):
        # infinite df -> normal distribution (0.0, 1.0, 0.0, 0.0)
        infinite_df = np.isposinf(df)

        mu = np.where(df > 1, 0.0, np.inf)

        condlist = ((df > 1) & (df <= 2), (df > 2) & np.isfinite(df), infinite_df)
        choicelist = (
            lambda df: np.broadcast_to(np.inf, df.shape),
            lambda df: df / (df - 2.0),
            lambda df: np.broadcast_to(1, df.shape),
        )
        mu2 = _lazyselect(condlist, choicelist, (df,), np.nan)

        g1 = np.where(df > 3, 0.0, np.nan)

        condlist = ((df > 2) & (df <= 4), (df > 4) & np.isfinite(df), infinite_df)
        choicelist = (
            lambda df: np.broadcast_to(np.inf, df.shape),
            lambda df: 6.0 / (df - 4.0),
            lambda df: np.broadcast_to(0, df.shape),
        )
        g2 = _lazyselect(condlist, choicelist, (df,), np.nan)

        return mu, mu2, g1, g2

    def _entropy(self, df):
        def regular(df):
            half = df / 2
            half1 = (df + 1) / 2
            return half1 * (sc.digamma(half1) - sc.digamma(half)) + np.log(
                np.sqrt(df) * sc.beta(half, 0.5)
            )

        def asymptotic(df):
            # Formula from Wolfram Alpha:
            # "asymptotic expansion (d+1)/2 * (digamma((d+1)/2) - digamma(d/2))
            #  + log(sqrt(d) * beta(d/2, 1/2))"
            h = (
                norm._entropy()
                + 1 / df
                + (df**-2.0) / 4
                - (df**-3.0) / 6
                - (df**-4.0) / 8
                + 3 / 10 * (df**-5.0)
                + (df**-6.0) / 4
            )
            return h

        return apply_where(df >= 100, df, asymptotic, regular)


t = t_gen(name="t")


class FitUniformFixedScaleDataError(FitDataError):
    """Raised when the data spread exceeds a fixed scale in ``uniform.fit``."""

    def __init__(self, ptp, fscale):
        # As in SciPy, ``args`` is set to a string rather than a 1-tuple, so it becomes a tuple
        # of characters.
        self.args = (
            "Invalid values in `data`.  Maximum likelihood estimation with "
            "the uniform distribution and fixed scale requires that "
            f"np.ptp(data) <= fscale, but np.ptp(data) = {ptp} and "
            f"fscale = {fscale}."
        )


class uniform_gen(rv_continuous):
    """A uniform continuous random variable on ``[loc, loc + scale]``."""

    def _rvs(self, size=None, random_state=None):
        return random_state.uniform(0.0, 1.0, size)

    def _pdf(self, x):
        return 1.0 * (x == x)

    def _cdf(self, x):
        return x

    def _ppf(self, q):
        return q

    def _stats(self):
        return 0.5, 1.0 / 12, 0, -1.2

    def _entropy(self):
        return 0.0

    def fit(self, data, *args, **kwds):
        """Maximum likelihood estimate for the location and scale parameters.

        The location is the minimum of the data and the scale its range. ``floc`` or ``fscale``
        fixes one; with a fixed scale larger than the range, the support is centered over the
        data. ``method='MM'`` uses the generic fit.
        """
        if _fit_method_is_mm(kwds):
            return super().fit(data, *args, **kwds)
        if len(args) > 0:
            raise TypeError("Too many arguments.")

        floc = kwds.pop("floc", None)
        fscale = kwds.pop("fscale", None)

        _remove_optimizer_parameters(kwds)

        if floc is not None and fscale is not None:
            # This check is for consistency with `rv_continuous.fit`.
            raise ValueError("All parameters fixed. There is nothing to optimize.")

        data = np.asarray(data)

        if not np.isfinite(data).all():
            raise ValueError("The data contains non-finite values.")

        # The log-likelihood is -n*log(scale) while loc <= x <= loc + scale for all x, so it is
        # maximized by the smallest scale that covers the data.
        if fscale is None:
            # scale is not fixed.
            if floc is None:
                # loc is not fixed, scale is not fixed.
                loc = data.min()
                scale = np.ptp(data)
            else:
                # loc is fixed, scale is not fixed.
                loc = floc
                scale = data.max() - loc
                if data.min() < loc:
                    raise FitDataError("uniform", lower=loc, upper=loc + scale)
        else:
            # loc is not fixed, scale is fixed.
            ptp = np.ptp(data)
            if ptp > fscale:
                raise FitUniformFixedScaleDataError(ptp=ptp, fscale=fscale)
            # If ptp < fscale, the ML estimate is not unique; center the support over the
            # interval [data.min(), data.max()].
            loc = data.min() - 0.5 * (fscale - ptp)
            scale = fscale

        # We expect the return values to be floating point, so ensure it
        # by explicitly converting to float.
        return float(loc), float(scale)


uniform = uniform_gen(a=0.0, b=1.0, name="uniform")

_distn_names = ["norm", "chi2", "expon", "f", "t", "uniform"]
_distn_gen_names = [name + "_gen" for name in _distn_names]

__all__ = _distn_names + _distn_gen_names
