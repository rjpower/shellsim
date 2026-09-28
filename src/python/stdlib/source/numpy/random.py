"""numpy.random: seeded pseudo-random numbers.

One bit generator, `PCG64` (a 128-bit LCG with XSL-RR output), backs every stream. A seed (an
int, a sequence of ints, or `None` for a fixed default, since shellsim has no host entropy) is
mixed into its state with splitmix64. Streams are deterministic and reproducible within
shellsim but do not match NumPy's; distributions, shapes, dtypes and call signatures do.

`Generator` (from `default_rng`) is the main interface. `RandomState` and the module-level
legacy functions (`seed`, `rand`, `randn`, ...) wrap a hidden `Generator`. `SeedSequence`,
`MT19937` and bit-generator state access are not provided.

Per-draw sampling runs in the native `_numpy_random` module. This file resolves arguments,
shapes and dtypes and broadcasts parameters.
"""

import numpy as _np
import _numpy_random as _nr

# ---------------------------------------------------------------------------
# Seeding: turn None, an int, or a sequence of ints into the u64 words `_pcg_seed` mixes with
# splitmix64.
# ---------------------------------------------------------------------------


def _is_plain_int(value):
    return isinstance(value, int) and not isinstance(value, bool)


def _int_to_u64_words(value):
    if value < 0:
        raise ValueError("expected a non-negative integer seed")
    if value == 0:
        return [0]
    words = []
    while value > 0:
        words.append(value & 0xFFFFFFFFFFFFFFFF)
        value >>= 64
    return words


def _seed_words(seed):
    if seed is None:
        return [0]
    if _is_plain_int(seed):
        return _int_to_u64_words(seed)
    try:
        items = list(seed)
    except TypeError:
        raise TypeError("seed must be None, an int, or a sequence of ints") from None
    words = []
    for item in items:
        if not _is_plain_int(item):
            raise TypeError("seed must be None, an int, or a sequence of ints")
        words.extend(_int_to_u64_words(item))
    return words or [0]


class PCG64:
    """PCG64 (O'Neill 2014, XSL-RR variant): shellsim's one bit generator for `numpy.random`.

    128-bit LCG state advanced by PCG's published multiplier, output through the xorshift-low
    random-rotate function that folds it to 64 bits (see the native `bitgen` module for the
    algorithm). Seeding expands `seed` (an int, a sequence of ints, or `None`) into words with
    `_seed_words` and mixes them with splitmix64; this does not reproduce NumPy's own `PCG64`
    streams (see this module's docstring).
    """

    def __init__(self, seed=None):
        self._state = _nr._pcg_seed(_seed_words(seed))


# ---------------------------------------------------------------------------
# Shared helpers.
# ---------------------------------------------------------------------------


def _shape_of(size):
    if size is None:
        return ()
    if isinstance(size, (int, _np.integer)):
        return (int(size),)
    return tuple(int(dim) for dim in size)


def _prod(shape):
    total = 1
    for dim in shape:
        total *= dim
    return total


def _scalar(values):
    return _np.asarray(values).reshape(-1)[0]


def _dtype_name(dtype):
    return _np.dtype(dtype).name


_INT_BOUNDS = {
    "bool": (0, 1),
    "int8": (-128, 127),
    "int16": (-32768, 32767),
    "int32": (-2147483648, 2147483647),
    "int64": (-9223372036854775808, 9223372036854775807),
    "uint8": (0, 255),
    "uint16": (0, 65535),
    "uint32": (0, 4294967295),
    "uint64": (0, 18446744073709551615),
}

_POISSON_LAM_MAX = 3074457345618258602.0


def _broadcast_params(size, params):
    """Broadcast float64 distribution parameters to the output shape.

    Mirrors NumPy's rule: with no explicit ``size``, the output shape is the parameters' own
    broadcast shape; with an explicit ``size``, the parameters must broadcast to it or the call
    fails with "shape mismatch" (NumPy's own wording for this case).
    """
    arrays = [_np.asarray(p, dtype=_np.float64) for p in params]
    param_shape = _np.broadcast_shapes(*(a.shape for a in arrays)) if arrays else ()
    shape = tuple(param_shape) if size is None else _shape_of(size)
    try:
        broadcast = [
            _np.ascontiguousarray(_np.broadcast_to(a, shape), dtype=_np.float64)
            for a in arrays
        ]
    except ValueError:
        raise ValueError("shape mismatch") from None
    return shape, broadcast


def _choice_population(a):
    if isinstance(a, (int, _np.integer)) and not isinstance(a, bool):
        pop_size = int(a)
        if pop_size <= 0:
            raise ValueError("a must be a positive integer unless no samples are taken")
        return pop_size, None
    a_arr = _np.asarray(a)
    if a_arr.shape[0] == 0:
        raise ValueError("'a' cannot be empty unless no samples are taken")
    return a_arr.shape[0], a_arr


def _choice_result(idx, values_source, size, shape):
    idx = _np.asarray(idx, dtype=_np.int64)
    result = idx if values_source is None else values_source[idx]
    if size is None:
        return _scalar(result)
    return _np.asarray(result).reshape(shape)


def _weighted_choice_with_replacement(raw_state, store_state, p_arr, n_samples):
    cumulative = _np.cumsum(p_arr)
    cumulative = cumulative / cumulative[-1]
    u, new_state = _nr._uniform01_fill(raw_state(), [n_samples], "float64")
    store_state(new_state)
    # u < 1 == cumulative[-1], and side="right" skips zero-weight entries.
    return _np.searchsorted(cumulative, u, side="right").astype(_np.int64)


def _weighted_choice_without_replacement(raw_state, store_state, p_arr, n_samples):
    if _np.count_nonzero(p_arr) < n_samples:
        raise ValueError("fewer non-zero probabilities than samples")
    u, new_state = _nr._uniform01_fill(raw_state(), [p_arr.shape[0]], "float64")
    store_state(new_state)
    # Efraimidis-Spirakis: ordering entries by decreasing log(1 - u) / p draws them one by one
    # with probability proportional to p among those not yet drawn.
    with _np.errstate(divide="ignore", invalid="ignore"):
        keys = _np.where(p_arr > 0, _np.log1p(-u) / p_arr, -_np.inf)
    return _np.argsort(-keys, kind="stable")[:n_samples].astype(_np.int64)


def _choice_indices(raw_state, store_state, pop_size, shape, replace, p, shuffle=True):
    n_samples = _prod(shape) if shape else 1
    p_arr = None
    if p is not None:
        p_arr = _np.asarray(p, dtype=_np.float64)
        if p_arr.shape[0] != pop_size:
            raise ValueError("a and p must have same size")
        if _np.any(p_arr < 0):
            raise ValueError("probabilities are not non-negative")
        if abs(float(_np.sum(p_arr)) - 1.0) > 1e-8:
            raise ValueError("probabilities do not sum to 1")
    if not replace:
        if n_samples > pop_size:
            raise ValueError(
                "Cannot take a larger sample than population when 'replace=False'"
            )
        if p_arr is None:
            idx, new_state = _nr._choice_without_replacement(
                raw_state(), pop_size, n_samples, shuffle
            )
            store_state(new_state)
            idx = _np.asarray(idx, dtype=_np.int64)
        else:
            idx = _weighted_choice_without_replacement(raw_state, store_state, p_arr, n_samples)
    else:
        if p_arr is None:
            low_b = _np.zeros(n_samples, dtype=_np.int64)
            high_incl_b = _np.full(n_samples, pop_size - 1, dtype=_np.int64)
            idx, new_state = _nr._bounded_int_fill(raw_state(), low_b, high_incl_b)
            store_state(new_state)
        else:
            idx = _weighted_choice_with_replacement(raw_state, store_state, p_arr, n_samples)
    return idx


def _shuffle(raw_state, store_state, x, axis=0):
    if isinstance(x, _np.ndarray):
        if not _np._writeable(x):
            raise ValueError("array is read-only")
        n = x.shape[axis]
        idx, new_state = _nr._shuffle_indices(raw_state(), n)
        store_state(new_state)
        x[...] = x[idx]
        return None
    n = len(x)
    idx, new_state = _nr._shuffle_indices(raw_state(), n)
    store_state(new_state)
    source = list(x)
    for i, value in enumerate(idx.tolist()):
        x[i] = source[value]
    return None


def _permutation(raw_state, store_state, x, axis=0):
    if isinstance(x, (int, _np.integer)) and not isinstance(x, bool):
        idx, new_state = _nr._shuffle_indices(raw_state(), int(x))
        store_state(new_state)
        return idx
    arr = _np.array(x, copy=True)
    _shuffle(raw_state, store_state, arr, axis=axis)
    return arr


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------


class Generator:
    """NumPy-style random number generator over a `PCG64` bit generator.

    See this module's docstring for shellsim's random-compatibility policy: shapes, dtypes, and
    statistics match NumPy; the underlying stream does not.
    """

    def __init__(self, bit_generator):
        self.bit_generator = bit_generator

    def _raw_state(self):
        return self.bit_generator._state

    def _store_state(self, new_state):
        self.bit_generator._state = new_state

    def random(self, size=None, dtype=_np.float64, out=None):
        dtype_name = _dtype_name(dtype)
        if dtype_name not in ("float64", "float32"):
            raise TypeError(f"Unsupported dtype {dtype_name} for random")
        shape = out.shape if out is not None else _shape_of(size)
        values, new_state = _nr._uniform01_fill(self._raw_state(), shape, dtype_name)
        self._store_state(new_state)
        if out is not None:
            out[...] = values
            return out
        return float(_scalar(values)) if shape == () else values

    def integers(self, low, high=None, size=None, dtype=_np.int64, endpoint=False):
        dtype_name = _dtype_name(dtype)
        if dtype_name not in _INT_BOUNDS:
            raise TypeError(f"Unsupported dtype {dtype_name} for integers")
        dmin, dmax = _INT_BOUNDS[dtype_name]
        if high is None:
            high = low
            low = 0
            if _np.any(_np.asarray(high) <= 0):
                raise ValueError("high <= 0")
        low_arr = _np.asarray(low, dtype=_np.int64)
        high_arr = _np.asarray(high, dtype=_np.int64)
        shape = _np.broadcast_shapes(low_arr.shape, high_arr.shape)
        target = shape if size is None else _shape_of(size)
        try:
            low_b = _np.ascontiguousarray(_np.broadcast_to(low_arr, target), dtype=_np.int64)
            high_b = _np.ascontiguousarray(_np.broadcast_to(high_arr, target), dtype=_np.int64)
        except ValueError:
            raise ValueError("shape mismatch") from None
        if _np.any(low_b < dmin):
            raise ValueError(f"low is out of bounds for {dtype_name}")
        high_incl = high_b if endpoint else high_b - 1
        if _np.any(high_incl > dmax):
            raise ValueError(f"high is out of bounds for {dtype_name}")
        if endpoint:
            if _np.any(low_b > high_b):
                raise ValueError("low > high")
        elif _np.any(low_b >= high_b):
            raise ValueError("low >= high")
        values, new_state = _nr._bounded_int_fill(self._raw_state(), low_b, high_incl)
        self._store_state(new_state)
        result = values if dtype_name == "int64" else values.astype(dtype_name)
        return _scalar(result) if target == () else result

    def choice(self, a, size=None, replace=True, p=None, shuffle=True):
        pop_size, values_source = _choice_population(a)
        shape = _shape_of(size)
        idx = _choice_indices(
            self._raw_state, self._store_state, pop_size, shape, replace, p, shuffle
        )
        return _choice_result(idx, values_source, size, shape)

    def shuffle(self, x, axis=0):
        return _shuffle(self._raw_state, self._store_state, x, axis=axis)

    def permutation(self, x, axis=0):
        return _permutation(self._raw_state, self._store_state, x, axis=axis)

    def uniform(self, low=0.0, high=1.0, size=None):
        shape, (low_a, high_a) = _broadcast_params(size, (low, high))
        width = high_a - low_a
        if not _np.all(_np.isfinite(width)):
            raise OverflowError("Range exceeds valid bounds")
        if _np.any(width < 0):
            raise ValueError("high - low < 0")
        base, new_state = _nr._uniform01_fill(self._raw_state(), shape, "float64")
        self._store_state(new_state)
        result = low_a + width * base
        return float(_scalar(result)) if shape == () else result

    def normal(self, loc=0.0, scale=1.0, size=None):
        shape, (loc_a, scale_a) = _broadcast_params(size, (loc, scale))
        if _np.any(scale_a < 0):
            raise ValueError("scale < 0")
        base, new_state = _nr._standard_normal_fill(self._raw_state(), shape, "float64")
        self._store_state(new_state)
        result = loc_a + scale_a * base
        return float(_scalar(result)) if shape == () else result

    def standard_normal(self, size=None, dtype=_np.float64, out=None):
        dtype_name = _dtype_name(dtype)
        if dtype_name not in ("float64", "float32"):
            raise TypeError(f"Unsupported dtype {dtype_name} for standard_normal")
        shape = out.shape if out is not None else _shape_of(size)
        values, new_state = _nr._standard_normal_fill(self._raw_state(), shape, dtype_name)
        self._store_state(new_state)
        if out is not None:
            out[...] = values
            return out
        return float(_scalar(values)) if shape == () else values

    def standard_exponential(self, size=None, dtype=_np.float64, method="zig", out=None):
        dtype_name = _dtype_name(dtype)
        if dtype_name not in ("float64", "float32"):
            raise TypeError(f"Unsupported dtype {dtype_name} for standard_exponential")
        if method not in ("zig", "inv"):
            raise ValueError("method must be 'zig' or 'inv'")
        shape = out.shape if out is not None else _shape_of(size)
        values, new_state = _nr._standard_exponential_fill(self._raw_state(), shape, dtype_name)
        self._store_state(new_state)
        if out is not None:
            out[...] = values
            return out
        return float(_scalar(values)) if shape == () else values

    def exponential(self, scale=1.0, size=None):
        shape, (scale_a,) = _broadcast_params(size, (scale,))
        if _np.any(scale_a < 0):
            raise ValueError("scale < 0")
        base, new_state = _nr._standard_exponential_fill(self._raw_state(), shape, "float64")
        self._store_state(new_state)
        result = scale_a * base
        return float(_scalar(result)) if shape == () else result

    def standard_gamma(self, shape, size=None, dtype=_np.float64, out=None):
        dtype_name = _dtype_name(dtype)
        if dtype_name not in ("float64", "float32"):
            raise TypeError(f"Unsupported dtype {dtype_name} for standard_gamma")
        target = out.shape if out is not None else size
        param_shape, (shape_a,) = _broadcast_params(target, (shape,))
        if _np.any(shape_a < 0):
            raise ValueError("shape < 0")
        values, new_state = _nr._standard_gamma_fill(self._raw_state(), shape_a, dtype_name)
        self._store_state(new_state)
        if out is not None:
            out[...] = values
            return out
        return float(_scalar(values)) if param_shape == () else values

    def gamma(self, shape, scale=1.0, size=None):
        param_shape, (shape_a, scale_a) = _broadcast_params(size, (shape, scale))
        if _np.any(shape_a < 0):
            raise ValueError("shape < 0")
        if _np.any(scale_a < 0):
            raise ValueError("scale < 0")
        values, new_state = _nr._standard_gamma_fill(self._raw_state(), shape_a, "float64")
        self._store_state(new_state)
        result = values * scale_a
        return float(_scalar(result)) if param_shape == () else result

    def beta(self, a, b, size=None):
        param_shape, (a_a, b_a) = _broadcast_params(size, (a, b))
        if _np.any(a_a <= 0):
            raise ValueError("a <= 0")
        if _np.any(b_a <= 0):
            raise ValueError("b <= 0")
        x, new_state = _nr._standard_gamma_fill(self._raw_state(), a_a, "float64")
        self._store_state(new_state)
        y, new_state = _nr._standard_gamma_fill(self._raw_state(), b_a, "float64")
        self._store_state(new_state)
        result = x / (x + y)
        return float(_scalar(result)) if param_shape == () else result

    def chisquare(self, df, size=None):
        param_shape, (df_a,) = _broadcast_params(size, (df,))
        if _np.any(df_a <= 0):
            raise ValueError("df <= 0")
        values, new_state = _nr._chisquare_fill(self._raw_state(), df_a)
        self._store_state(new_state)
        return float(_scalar(values)) if param_shape == () else values

    def f(self, dfnum, dfden, size=None):
        param_shape, (dfnum_a, dfden_a) = _broadcast_params(size, (dfnum, dfden))
        if _np.any(dfnum_a <= 0):
            raise ValueError("dfnum <= 0")
        if _np.any(dfden_a <= 0):
            raise ValueError("dfden <= 0")
        values, new_state = _nr._f_fill(self._raw_state(), dfnum_a, dfden_a)
        self._store_state(new_state)
        return float(_scalar(values)) if param_shape == () else values

    def standard_t(self, df, size=None):
        param_shape, (df_a,) = _broadcast_params(size, (df,))
        if _np.any(df_a <= 0):
            raise ValueError("df <= 0")
        values, new_state = _nr._standard_t_fill(self._raw_state(), df_a)
        self._store_state(new_state)
        return float(_scalar(values)) if param_shape == () else values

    def lognormal(self, mean=0.0, sigma=1.0, size=None):
        shape, (mean_a, sigma_a) = _broadcast_params(size, (mean, sigma))
        if _np.any(sigma_a < 0):
            raise ValueError("sigma < 0")
        base, new_state = _nr._standard_normal_fill(self._raw_state(), shape, "float64")
        self._store_state(new_state)
        result = _np.exp(mean_a + sigma_a * base)
        return float(_scalar(result)) if shape == () else result

    def binomial(self, n, p, size=None):
        n_scalar = _np.ndim(n) == 0
        p_scalar = _np.ndim(p) == 0
        n_in = _np.asarray(n)
        if n_in.dtype.kind not in "iub":
            raise TypeError(
                f"Cannot cast array data from dtype('{n_in.dtype.name}') "
                "to dtype('int64') according to the rule 'safe'"
            )
        n_arr = n_in.astype(_np.int64)
        p_arr = _np.asarray(p, dtype=_np.float64)
        shape = _np.broadcast_shapes(n_arr.shape, p_arr.shape)
        target = shape if size is None else _shape_of(size)
        try:
            n_b = _np.ascontiguousarray(_np.broadcast_to(n_arr, target), dtype=_np.int64)
            p_b = _np.ascontiguousarray(_np.broadcast_to(p_arr, target), dtype=_np.float64)
        except ValueError:
            raise ValueError("shape mismatch") from None
        if _np.any(n_b < 0):
            raise ValueError("n < 0")
        nan_msg = "p < 0, p > 1 or p is NaN" if p_scalar else "p < 0, p > 1 or p contains NaNs"
        if not _np.all((p_b >= 0) & (p_b <= 1)):
            raise ValueError(nan_msg)
        values, new_state = _nr._binomial_fill(self._raw_state(), n_b, p_b)
        self._store_state(new_state)
        return int(_scalar(values)) if target == () else values

    def poisson(self, lam=1.0, size=None):
        lam_arr = _np.asarray(lam, dtype=_np.float64)
        target = lam_arr.shape if size is None else _shape_of(size)
        try:
            lam_b = _np.ascontiguousarray(_np.broadcast_to(lam_arr, target), dtype=_np.float64)
        except ValueError:
            raise ValueError("shape mismatch") from None
        if not _np.all(lam_b < _POISSON_LAM_MAX):
            raise ValueError("lam value too large")
        if _np.any(lam_b < 0):
            raise ValueError("lam < 0 or lam is NaN")
        values, new_state = _nr._poisson_fill(self._raw_state(), lam_b)
        self._store_state(new_state)
        return int(_scalar(values)) if target == () else values


def default_rng(seed=None):
    """Return a `Generator` seeded from `seed`, backed by `PCG64` (NumPy's own default bit
    generator since 1.17). `seed` may be `None`, an int, a sequence of ints, an existing
    `Generator` (returned unchanged), or an existing `PCG64`.
    """
    if isinstance(seed, Generator):
        return seed
    if isinstance(seed, PCG64):
        return Generator(seed)
    return Generator(PCG64(seed))


# ---------------------------------------------------------------------------
# RandomState (legacy) and the module-level functions built on it.
# ---------------------------------------------------------------------------


class RandomState:
    """Legacy `numpy.random` interface (`rand`, `randn`, `randint`, ...), kept for older call
    patterns and for SciPy's `check_random_state`. This is a thin wrapper around one `Generator`;
    shellsim's `RandomState` does not reproduce NumPy's own legacy generator's stream (see this
    module's docstring).
    """

    def __init__(self, seed=None):
        self.seed(seed)

    def seed(self, seed=None):
        self._gen = Generator(PCG64(seed))

    def rand(self, *shape):
        return self._gen.random(shape if shape else None)

    def randn(self, *shape):
        return self._gen.standard_normal(shape if shape else None)

    def random_sample(self, size=None):
        return self._gen.random(size)

    random = random_sample

    def randint(self, low, high=None, size=None, dtype=_np.int64):
        # `Generator.integers` returns a NumPy scalar for a scalar draw; legacy `randint` returns
        # a plain Python `int` instead, matching NumPy's own `RandomState`.
        result = self._gen.integers(low, high, size=size, dtype=dtype)
        return int(result) if _np.ndim(result) == 0 else result

    def choice(self, a, size=None, replace=True, p=None):
        return self._gen.choice(a, size, replace, p)

    def shuffle(self, x):
        return self._gen.shuffle(x)

    def permutation(self, x):
        return self._gen.permutation(x)

    def uniform(self, low=0.0, high=1.0, size=None):
        return self._gen.uniform(low, high, size)

    def normal(self, loc=0.0, scale=1.0, size=None):
        return self._gen.normal(loc, scale, size)

    def standard_normal(self, size=None):
        return self._gen.standard_normal(size)

    def standard_exponential(self, size=None):
        return self._gen.standard_exponential(size)

    def exponential(self, scale=1.0, size=None):
        return self._gen.exponential(scale, size)

    def standard_gamma(self, shape, size=None):
        return self._gen.standard_gamma(shape, size)

    def gamma(self, shape, scale=1.0, size=None):
        return self._gen.gamma(shape, scale, size)

    def chisquare(self, df, size=None):
        return self._gen.chisquare(df, size)

    def f(self, dfnum, dfden, size=None):
        return self._gen.f(dfnum, dfden, size)

    def standard_t(self, df, size=None):
        return self._gen.standard_t(df, size)

    def binomial(self, n, p, size=None):
        return self._gen.binomial(n, p, size)

    def poisson(self, lam=1.0, size=None):
        return self._gen.poisson(lam, size)


_rand = RandomState()


def seed(seed=None):
    _rand.seed(seed)


def rand(*shape):
    return _rand.rand(*shape)


def randn(*shape):
    return _rand.randn(*shape)


def randint(low, high=None, size=None, dtype=_np.int64):
    return _rand.randint(low, high, size, dtype)


def random_sample(size=None):
    return _rand.random_sample(size)


def random(size=None):
    return _rand.random_sample(size)


def choice(a, size=None, replace=True, p=None):
    return _rand.choice(a, size, replace, p)


def shuffle(x):
    return _rand.shuffle(x)


def permutation(x):
    return _rand.permutation(x)


def uniform(low=0.0, high=1.0, size=None):
    return _rand.uniform(low, high, size)


def normal(loc=0.0, scale=1.0, size=None):
    return _rand.normal(loc, scale, size)


def standard_normal(size=None):
    return _rand.standard_normal(size)


def standard_exponential(size=None):
    return _rand.standard_exponential(size)


def exponential(scale=1.0, size=None):
    return _rand.exponential(scale, size)


def standard_gamma(shape, size=None):
    return _rand.standard_gamma(shape, size)


def gamma(shape, scale=1.0, size=None):
    return _rand.gamma(shape, scale, size)


def chisquare(df, size=None):
    return _rand.chisquare(df, size)


def f(dfnum, dfden, size=None):
    return _rand.f(dfnum, dfden, size)


def standard_t(df, size=None):
    return _rand.standard_t(df, size)


def binomial(n, p, size=None):
    return _rand.binomial(n, p, size)


def poisson(lam=1.0, size=None):
    return _rand.poisson(lam, size)


class _Mtrand:
    """Minimal shim for `np.random.mtrand._rand`, which SciPy's `check_random_state` reads
    directly for `random_state=None` (real NumPy's `numpy.random` package always has a `mtrand`
    submodule); shellsim implements nothing else of it.
    """


mtrand = _Mtrand()
mtrand._rand = _rand
