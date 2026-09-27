"""Discrete distributions, following SciPy 1.18's ``scipy/stats/_discrete_distns.py``.

shellsim implements ``binom`` and ``poisson``. ``binom`` evaluates through SciPy's private
Boost-based ufuncs in ``scipy.special._ufuncs``, and ``poisson`` through the Cephes
``pdtr`` family, as SciPy does.
"""

import numpy as np
import scipy.special._ufuncs as scu
from numpy import ceil, exp, floor, sqrt
from scipy import special
from scipy._lib._util import apply_where
from scipy.special import gammaln as gamln
from scipy.stats._distn_infrastructure import _isintegral, rv_discrete


class binom_gen(rv_discrete):
    """A binomial discrete random variable: successes in ``n`` trials of probability ``p``."""

    def _rvs(self, n, p, size=None, random_state=None):
        if not np.all(n == np.floor(n)):
            raise ValueError("`n` must be integral.")
        return random_state.binomial(np.asarray(n, dtype=int), p, size)

    def _argcheck(self, n, p):
        return (n >= 0) & _isintegral(n) & (p >= 0) & (p <= 1)

    def _get_support(self, n, p):
        return self.a, n

    def _logpmf(self, x, n, p):
        k = floor(x)
        combiln = gamln(n + 1) - (gamln(k + 1) + gamln(n - k + 1))
        return combiln + special.xlogy(k, p) + special.xlog1py(n - k, -p)

    def _pmf(self, x, n, p):
        # binom.pmf(k) = choose(n, k) * p**k * (1-p)**(n-k)
        return scu._binom_pmf(x, n, p)

    def _cdf(self, x, n, p):
        k = floor(x)
        return scu._binom_cdf(k, n, p)

    def _sf(self, x, n, p):
        k = floor(x)
        return scu._binom_sf(k, n, p)

    def _isf(self, x, n, p):
        return scu._binom_isf(x, n, p)

    def _ppf(self, q, n, p):
        return scu._binom_ppf(q, n, p)

    def _stats(self, n, p, moments="mv"):
        mu = n * p
        var = mu - n * np.square(p)
        g1, g2 = None, None
        if "s" in moments:
            pq = p - np.square(p)
            npq_sqrt = np.sqrt(n * pq)
            t1 = np.reciprocal(npq_sqrt)
            t2 = (2.0 * p) / npq_sqrt
            g1 = t1 - t2
        if "k" in moments:
            pq = p - np.square(p)
            npq = n * pq
            t1 = np.reciprocal(npq)
            t2 = 6.0 / n
            g2 = t1 - t2
        return mu, var, g1, g2


binom = binom_gen(name="binom")


class poisson_gen(rv_discrete):
    """A Poisson discrete random variable with mean ``mu``."""

    # Override rv_discrete._argcheck to allow mu=0.
    def _argcheck(self, mu):
        return mu >= 0

    def _rvs(self, mu, size=None, random_state=None):
        return random_state.poisson(mu, size)

    def _logpmf(self, k, mu):
        Pk = special.xlogy(k, mu) - gamln(k + 1) - mu
        return Pk

    def _pmf(self, k, mu):
        # poisson.pmf(k) = exp(-mu) * mu**k / k!
        return exp(self._logpmf(k, mu))

    def _cdf(self, x, mu):
        k = floor(x)
        return special.pdtr(k, mu)

    def _sf(self, x, mu):
        k = floor(x)
        return special.pdtrc(k, mu)

    def _ppf(self, q, mu):
        vals = ceil(special.pdtrik(q, mu))
        vals1 = np.maximum(vals - 1, 0)
        temp = special.pdtr(vals1, mu)
        return np.where(temp >= q, vals1, vals)

    def _stats(self, mu):
        var = mu
        tmp = np.asarray(mu)
        mu_nonzero = tmp > 0
        g1 = apply_where(mu_nonzero, tmp, lambda x: sqrt(1.0 / x), fill_value=np.inf)
        g2 = apply_where(mu_nonzero, tmp, lambda x: 1.0 / x, fill_value=np.inf)
        return mu, var, g1, g2


poisson = poisson_gen(name="poisson", longname="A Poisson")

_distn_names = ["binom", "poisson"]
_distn_gen_names = [name + "_gen" for name in _distn_names]

__all__ = _distn_names + _distn_gen_names
