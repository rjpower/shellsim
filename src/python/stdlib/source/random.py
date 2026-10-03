"""Pseudo-random numbers from a deterministic 64-bit generator.

The core generator is a splitmix64 sequence over Python integers, so results are identical on
every host and never depend on host entropy. ``seed(None)`` draws its seed from the virtual
clock and process id. ``SystemRandom`` reads ``os.urandom``, which is also modeled.
"""

import math as _math
import os as _os
import time as _time

__all__ = [
    "Random", "SystemRandom", "betavariate", "binomialvariate", "choice", "choices",
    "expovariate", "gammavariate", "gauss", "getrandbits", "getstate", "lognormvariate",
    "normalvariate", "paretovariate", "randbytes", "randint", "random", "randrange", "sample",
    "seed", "setstate", "shuffle", "triangular", "uniform", "vonmisesvariate", "weibullvariate",
]

NV_MAGICCONST = 4 * _math.exp(-0.5) / _math.sqrt(2.0)
TWOPI = 2.0 * _math.pi
LOG4 = _math.log(4.0)
SG_MAGICCONST = 1.0 + _math.log(4.5)
BPF = 53
RECIP_BPF = 2 ** -BPF

_MASK64 = (1 << 64) - 1


def _mix64(value):
    """The splitmix64 output function: a bijective scramble of one 64-bit word."""
    value = (value ^ (value >> 30)) * 0xBF58476D1CE4E5B9 & _MASK64
    value = (value ^ (value >> 27)) * 0x94D049BB133111EB & _MASK64
    return value ^ (value >> 31)


class Random:
    """A seedable generator; subclasses may override ``random``, ``seed``, ``getstate``,
    ``setstate`` and ``getrandbits`` to supply another source of bits."""

    VERSION = 3

    def __init__(self, x=None):
        self.seed(x)
        self.gauss_next = None

    def seed(self, a=None, version=2):
        if a is None:
            a = _time.time_ns() ^ (_os.getpid() << 32)
        elif isinstance(a, (str, bytes, bytearray)):
            if version == 1 and isinstance(a, str):
                a = hash(a)
            else:
                import hashlib

                if isinstance(a, str):
                    a = a.encode()
                a = int.from_bytes(bytes(a) + hashlib.sha512(a).digest(), "big")
        elif isinstance(a, float):
            a = hash(a)
        elif not isinstance(a, int):
            if version == 1:
                a = hash(a)
            else:
                raise TypeError(
                    "The only supported seed types are:\nNone, int, float, str, bytes, "
                    "and bytearray."
                )
        self._state = _mix64(abs(a) & _MASK64) ^ ((abs(a) >> 64) & _MASK64)
        self.gauss_next = None

    def _next64(self):
        self._state = (self._state + 0x9E3779B97F4A7C15) & _MASK64
        return _mix64(self._state)

    def random(self):
        """The next float in [0.0, 1.0) with 53 random bits."""
        return (self._next64() >> 11) * RECIP_BPF

    def getrandbits(self, k):
        if k < 0:
            raise ValueError("number of bits must be non-negative")
        if k == 0:
            return 0
        result = 0
        produced = 0
        while produced < k:
            result = (result << 64) | self._next64()
            produced += 64
        return result >> (produced - k)

    def getstate(self):
        return self.VERSION, self._state, self.gauss_next

    def setstate(self, state):
        version = state[0]
        if version != self.VERSION:
            raise ValueError(
                "state with version %s passed to Random.setstate() of version %s"
                % (version, self.VERSION)
            )
        self._state = state[1] & _MASK64
        self.gauss_next = state[2]

    def __getstate__(self):
        return self.getstate()

    def __setstate__(self, state):
        self.setstate(state)

    def __reduce__(self):
        return self.__class__, (), self.getstate()

    def _randbelow(self, n):
        """An integer in [0, n) without modulo bias."""
        if n <= 0:
            raise ValueError("n must be positive")
        k = n.bit_length()
        r = self.getrandbits(k)
        while r >= n:
            r = self.getrandbits(k)
        return r

    def randbytes(self, n):
        if n < 0:
            raise ValueError("number of bytes must be non-negative")
        return self.getrandbits(n * 8).to_bytes(n, "little")

    def randrange(self, start, stop=None, step=1):
        istart = _index(start)
        if stop is None:
            if step != 1:
                raise TypeError("Missing a non-None stop argument")
            if istart > 0:
                return self._randbelow(istart)
            raise ValueError("empty range for randrange()")
        istop = _index(stop)
        width = istop - istart
        istep = _index(step)
        if istep == 1:
            if width > 0:
                return istart + self._randbelow(width)
            raise ValueError("empty range in randrange(%d, %d)" % (istart, istop))
        if istep > 0:
            n = (width + istep - 1) // istep
        elif istep < 0:
            n = (width + istep + 1) // istep
        else:
            raise ValueError("zero step for randrange()")
        if n <= 0:
            raise ValueError("empty range in randrange(%d, %d, %d)" % (istart, istop, istep))
        return istart + istep * self._randbelow(n)

    def randint(self, a, b):
        return self.randrange(a, b + 1)

    def choice(self, seq):
        if not len(seq):
            raise IndexError("Cannot choose from an empty sequence")
        return seq[self._randbelow(len(seq))]

    def shuffle(self, x):
        for i in reversed(range(1, len(x))):
            j = self._randbelow(i + 1)
            x[i], x[j] = x[j], x[i]

    def sample(self, population, k, *, counts=None):
        if not isinstance(population, (list, tuple, range, str, bytes, bytearray)):
            if isinstance(population, (set, frozenset, dict)):
                raise TypeError("Population must be a sequence.  For dicts or sets, use sorted(d).")
            population = list(population)
        n = len(population)
        if counts is not None:
            cum_counts = []
            total = 0
            for count in counts:
                total += count
                cum_counts.append(total)
            if len(cum_counts) != n:
                raise ValueError("The number of counts does not match the population")
            if total <= 0:
                raise ValueError("Total of counts must be greater than zero")
            selections = self.sample(range(total), k=k)
            import bisect

            return [population[bisect.bisect(cum_counts, s)] for s in selections]
        if not 0 <= k <= n:
            raise ValueError("Sample larger than population or is negative")
        result = [None] * k
        setsize = 21
        if k > 5:
            setsize += 4 ** _math.ceil(_math.log(k * 3, 4))
        if n <= setsize:
            pool = list(population)
            for i in range(k):
                j = self._randbelow(n - i)
                result[i] = pool[j]
                pool[j] = pool[n - i - 1]
        else:
            selected = set()
            for i in range(k):
                j = self._randbelow(n)
                while j in selected:
                    j = self._randbelow(n)
                selected.add(j)
                result[i] = population[j]
        return result

    def choices(self, population, weights=None, *, cum_weights=None, k=1):
        n = len(population)
        if cum_weights is None:
            if weights is None:
                floor = _math.floor
                n += 0.0
                return [population[floor(self.random() * n)] for _ in range(k)]
            cum_weights = []
            total = 0
            for weight in weights:
                total += weight
                cum_weights.append(total)
        elif weights is not None:
            raise TypeError("Cannot specify both weights and cumulative weights")
        if len(cum_weights) != n:
            raise ValueError("The number of weights does not match the population")
        total = cum_weights[-1] + 0.0
        if total <= 0.0:
            raise ValueError("Total of weights must be greater than zero")
        if not _math.isfinite(total):
            raise ValueError("Total of weights must be finite")
        import bisect

        hi = n - 1
        return [population[bisect.bisect(cum_weights, self.random() * total, 0, hi)] for _ in range(k)]

    def uniform(self, a, b):
        return a + (b - a) * self.random()

    def triangular(self, low=0.0, high=1.0, mode=None):
        u = self.random()
        try:
            c = 0.5 if mode is None else (mode - low) / (high - low)
        except ZeroDivisionError:
            return low
        if u > c:
            u = 1.0 - u
            c = 1.0 - c
            low, high = high, low
        return low + (high - low) * _math.sqrt(u * c)

    def normalvariate(self, mu=0.0, sigma=1.0):
        while True:
            u1 = self.random()
            u2 = 1.0 - self.random()
            z = NV_MAGICCONST * (u1 - 0.5) / u2
            zz = z * z / 4.0
            if zz <= -_math.log(u2):
                break
        return mu + z * sigma

    def gauss(self, mu=0.0, sigma=1.0):
        z = self.gauss_next
        self.gauss_next = None
        if z is None:
            x2pi = self.random() * TWOPI
            g2rad = _math.sqrt(-2.0 * _math.log(1.0 - self.random()))
            z = _math.cos(x2pi) * g2rad
            self.gauss_next = _math.sin(x2pi) * g2rad
        return mu + z * sigma

    def lognormvariate(self, mu, sigma):
        return _math.exp(self.normalvariate(mu, sigma))

    def expovariate(self, lambd=1.0):
        return -_math.log(1.0 - self.random()) / lambd

    def vonmisesvariate(self, mu, kappa):
        if kappa <= 1e-6:
            return TWOPI * self.random()
        s = 0.5 / kappa
        r = s + _math.sqrt(1.0 + s * s)
        while True:
            u1 = self.random()
            z = _math.cos(_math.pi * u1)
            d = z / (r + z)
            u2 = self.random()
            if u2 < 1.0 - d * d or u2 <= (1.0 - d) * _math.exp(d):
                break
        q = 1.0 / r
        f = (q + z) / (1.0 + q * z)
        u3 = self.random()
        if u3 > 0.5:
            theta = (mu + _math.acos(f)) % TWOPI
        else:
            theta = (mu - _math.acos(f)) % TWOPI
        return theta

    def gammavariate(self, alpha, beta):
        if alpha <= 0.0 or beta <= 0.0:
            raise ValueError("gammavariate: alpha and beta must be > 0.0")
        if alpha > 1.0:
            ainv = _math.sqrt(2.0 * alpha - 1.0)
            bbb = alpha - LOG4
            ccc = alpha + ainv
            while True:
                u1 = self.random()
                if not 1e-7 < u1 < 0.9999999:
                    continue
                u2 = 1.0 - self.random()
                v = _math.log(u1 / (1.0 - u1)) / ainv
                x = alpha * _math.exp(v)
                z = u1 * u1 * u2
                r = bbb + ccc * v - x
                if r + SG_MAGICCONST - 4.5 * z >= 0.0 or r >= _math.log(z):
                    return x * beta
        elif alpha == 1.0:
            return -_math.log(1.0 - self.random()) * beta
        else:
            while True:
                u = self.random()
                b = (_math.e + alpha) / _math.e
                p = b * u
                if p <= 1.0:
                    x = p ** (1.0 / alpha)
                else:
                    x = -_math.log((b - p) / alpha)
                u1 = self.random()
                if p > 1.0:
                    if u1 <= x ** (alpha - 1.0):
                        break
                elif u1 <= _math.exp(-x):
                    break
            return x * beta

    def betavariate(self, alpha, beta):
        y = self.gammavariate(alpha, 1.0)
        if y:
            return y / (y + self.gammavariate(beta, 1.0))
        return 0.0

    def paretovariate(self, alpha):
        u = 1.0 - self.random()
        return u ** (-1.0 / alpha)

    def weibullvariate(self, alpha, beta):
        u = 1.0 - self.random()
        return alpha * (-_math.log(u)) ** (1.0 / beta)

    def binomialvariate(self, n=1, p=0.5):
        if n < 0:
            raise ValueError("n must be non-negative")
        if p <= 0.0 or p >= 1.0:
            if p == 0.0:
                return 0
            if p == 1.0:
                return n
            raise ValueError("p must be in the range 0.0 <= p <= 1.0")
        # Direct simulation keeps the cost proportional to n, which the VM meters.
        successes = 0
        for _ in range(n):
            if self.random() < p:
                successes += 1
        return successes


def _index(value):
    if isinstance(value, int):
        return int(value)
    index = getattr(type(value), "__index__", None)
    if index is None:
        raise TypeError("'%s' object cannot be interpreted as an integer" % type(value).__name__)
    return index(value)


class SystemRandom(Random):
    """A generator fed by ``os.urandom``; it cannot be seeded or have its state saved."""

    def random(self):
        return (int.from_bytes(_os.urandom(7), "big") >> 3) * RECIP_BPF

    def getrandbits(self, k):
        if k < 0:
            raise ValueError("number of bits must be non-negative")
        numbytes = (k + 7) // 8
        x = int.from_bytes(_os.urandom(numbytes), "big")
        return x >> (numbytes * 8 - k)

    def randbytes(self, n):
        return _os.urandom(n)

    def seed(self, *args, **kwds):
        return None

    def _notimplemented(self, *args, **kwds):
        raise NotImplementedError("System entropy source does not have state.")

    getstate = setstate = _notimplemented


_inst = Random()
seed = _inst.seed
random = _inst.random
uniform = _inst.uniform
triangular = _inst.triangular
randint = _inst.randint
choice = _inst.choice
randrange = _inst.randrange
sample = _inst.sample
shuffle = _inst.shuffle
choices = _inst.choices
normalvariate = _inst.normalvariate
lognormvariate = _inst.lognormvariate
expovariate = _inst.expovariate
vonmisesvariate = _inst.vonmisesvariate
gammavariate = _inst.gammavariate
gauss = _inst.gauss
betavariate = _inst.betavariate
binomialvariate = _inst.binomialvariate
paretovariate = _inst.paretovariate
weibullvariate = _inst.weibullvariate
getstate = _inst.getstate
setstate = _inst.setstate
getrandbits = _inst.getrandbits
randbytes = _inst.randbytes
