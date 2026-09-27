"""``numpy.random``: NumPy 2.5's seeded random streams.

This ports the parts of ``numpy/random`` that ordinary code uses from ``bit_generator.pyx``,
``_mt19937.pyx``, ``_pcg64.pyx``, ``_generator.pyx``, ``mtrand.pyx`` and
``_bounded_integers.pyx.in``: ``SeedSequence``, the ``MT19937`` and ``PCG64`` bit generators,
``Generator`` (``random``, ``integers``, ``uniform``, ``standard_normal``, ``normal``,
``choice``, ``shuffle`` and ``permutation``), and the legacy ``RandomState`` with the
module-level functions bound to its global instance. Seeding and every draw run in the native
module ``_numpy_random``, so seeded streams match NumPy bit for bit. Each bit generator keeps
its state in a ``uint64`` array, ``_state``, whose layout the native module defines.

shellsim programs have no host entropy, so an unseeded ``SeedSequence`` uses the fixed entropy
``_FIXED_ENTROPY``; ``default_rng()``, ``RandomState()`` and the global legacy stream are
therefore reproducible. Distributions other than these are not provided.
"""

import operator

import numpy as np
import _numpy_random as _native
from _numpy_shape import _normalize_axis_index as normalize_axis_index

__all__ = [
    "BitGenerator",
    "Generator",
    "MT19937",
    "PCG64",
    "RandomState",
    "SeedSequence",
    "choice",
    "default_rng",
    "get_bit_generator",
    "get_state",
    "normal",
    "permutation",
    "rand",
    "randint",
    "randn",
    "random",
    "random_integers",
    "random_sample",
    "ranf",
    "sample",
    "seed",
    "set_bit_generator",
    "set_state",
    "shuffle",
    "standard_normal",
    "uniform",
]

DEFAULT_POOL_SIZE = 4
# Stands in for `secrets.randbits(128)`: the first 128 fraction bits of pi.
_FIXED_ENTROPY = 0x243F6A8885A308D313198A2E03707344
_MASK32 = 0xFFFFFFFF
_RK_STATE_LEN = 624
_INTEGER_DTYPES = ("int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64", "bool")


def _normalize_size(size):
    """The shape and element count ``np.empty(size)`` would allocate."""
    try:
        shape = (operator.index(size),)
    except TypeError:
        shape = tuple(operator.index(dimension) for dimension in size)
    count = 1
    for dimension in shape:
        if dimension < 0:
            raise ValueError("negative dimensions are not allowed")
        count *= dimension
    return shape, count


def _int_to_uint32_array(n):
    words = []
    if n < 0:
        raise ValueError("expected non-negative integer")
    if n == 0:
        words.append(0)
    n = int(n)
    while n > 0:
        words.append(n & _MASK32)
        n //= 2**32
    return np.array(words, dtype=np.uint32)


def _coerce_to_uint32_array(x):
    if isinstance(x, np.ndarray) and x.dtype == np.dtype(np.uint32):
        return x.copy()
    elif isinstance(x, str):
        if x.startswith("0x"):
            x = int(x, base=16)
        elif x[:1] and "0" <= x[:1] <= "9":
            x = int(x)
        else:
            raise ValueError("unrecognized seed string")
    if isinstance(x, (int, np.integer)):
        return _int_to_uint32_array(x)
    elif isinstance(x, (float, np.inexact)):
        raise TypeError("seed must be integer")
    else:
        if len(x) == 0:
            return np.array([], dtype=np.uint32)
        # Should be a sequence of interpretable-as-ints. Convert each one to a uint32 array and
        # concatenate.
        subseqs = []
        for v in x:
            if hasattr(v, "__len__") and not isinstance(v, str):
                raise TypeError("SeedSequence does not accept nested sequences.")
            subseqs.append(_coerce_to_uint32_array(v))
        return np.concatenate(subseqs)


class SeedSequence:
    """Mixes seed entropy into well-distributed initial states for bit generators."""

    def __init__(
        self, entropy=None, *, spawn_key=(), pool_size=DEFAULT_POOL_SIZE, n_children_spawned=0
    ):
        if pool_size < DEFAULT_POOL_SIZE:
            raise ValueError(
                f"The size of the entropy pool should be at least {DEFAULT_POOL_SIZE}"
            )
        if entropy is None:
            entropy = _FIXED_ENTROPY
        elif not isinstance(entropy, (int, np.integer, list, tuple, range, np.ndarray)):
            raise TypeError(
                f"SeedSequence expects int or sequence of ints for entropy not {entropy}"
            )
        self.entropy = entropy
        self.spawn_key = tuple(spawn_key)
        self.pool_size = pool_size
        self.n_children_spawned = n_children_spawned
        self.pool = _native.seed_sequence_pool(self._assembled_entropy(), pool_size)

    def __repr__(self):
        lines = [f"{type(self).__name__}(", f"    entropy={self.entropy!r},"]
        # Omit some entries if they are left as the defaults in order to simplify things.
        if self.spawn_key:
            lines.append(f"    spawn_key={self.spawn_key!r},")
        if self.pool_size != DEFAULT_POOL_SIZE:
            lines.append(f"    pool_size={self.pool_size!r},")
        if self.n_children_spawned != 0:
            lines.append(f"    n_children_spawned={self.n_children_spawned!r},")
        lines.append(")")
        return "\n".join(lines)

    @property
    def state(self):
        return {
            k: getattr(self, k)
            for k in ["entropy", "spawn_key", "pool_size", "n_children_spawned"]
            if getattr(self, k) is not None
        }

    def _assembled_entropy(self):
        run_entropy = _coerce_to_uint32_array(self.entropy)
        spawn_entropy = _coerce_to_uint32_array(self.spawn_key)
        if len(spawn_entropy) > 0 and len(run_entropy) < self.pool_size:
            # Pad so spawned children never collide with a longer run entropy.
            diff = self.pool_size - len(run_entropy)
            run_entropy = np.concatenate([run_entropy, np.zeros(diff, dtype=np.uint32)])
        return np.concatenate([run_entropy, spawn_entropy])

    def generate_state(self, n_words, dtype=np.uint32):
        """Return ``n_words`` of seed material as ``uint32``, or ``uint64`` pairs."""
        out_dtype = np.dtype(dtype)
        if out_dtype == np.dtype(np.uint32):
            wide = False
        elif out_dtype == np.dtype(np.uint64):
            wide = True
        else:
            raise ValueError("only support uint32 or uint64")
        return _native.seed_sequence_generate(self.pool, operator.index(n_words), wide)

    def spawn(self, n_children):
        """Create ``n_children`` independent child sequences."""
        if n_children < 0:
            raise ValueError("n_children must be non-negative")
        seqs = []
        for i in range(self.n_children_spawned, self.n_children_spawned + n_children):
            seqs.append(
                type(self)(self.entropy, spawn_key=self.spawn_key + (i,), pool_size=self.pool_size)
            )
        self.n_children_spawned += n_children
        return seqs


class BitGenerator:
    """Base class of the bit generators; subclasses set ``_state``."""

    def __init__(self, seed=None):
        if type(self) is BitGenerator:
            raise NotImplementedError("BitGenerator is a base class and cannot be instantized")
        if not isinstance(seed, SeedSequence):
            seed = SeedSequence(seed)
        self._seed_seq = seed

    @property
    def state(self):
        raise NotImplementedError("Not implemented in base BitGenerator")

    @property
    def seed_seq(self):
        return self._seed_seq

    def spawn(self, n_children):
        """Create new independent child bit generators."""
        if n_children < 0:
            raise ValueError("n_children must be non-negative")
        if not isinstance(self._seed_seq, SeedSequence):
            raise TypeError("The underlying SeedSequence does not implement spawning.")
        return [type(self)(seed=s) for s in self._seed_seq.spawn(n_children)]

    def random_raw(self, size=None, output=True):
        """Return the generator's raw outputs as ``uint64``."""
        if size is None:
            value = _native.random_raw(self._state, 1)
            return int(value[0]) if output else None
        shape, count = _normalize_size(size)
        values = _native.random_raw(self._state, count)
        return values.reshape(shape) if output else None


class MT19937(BitGenerator):
    """The Mersenne Twister bit generator, with NumPy's legacy seeding."""

    def __init__(self, seed=None):
        BitGenerator.__init__(self, seed)
        key = self._seed_seq.generate_state(_RK_STATE_LEN, np.uint32)
        # MSB is 1; assuring non-zero initial array.
        key[0] = 0x80000000
        self._state = _native.mt19937_from_key(key, _RK_STATE_LEN - 1)

    def _legacy_seeding(self, seed):
        try:
            if seed is None:
                seed = SeedSequence()
                key = seed.generate_state(_RK_STATE_LEN)
                key[0] = 0x80000000
                # As in NumPy, this path keeps the previous position.
                self._state = _native.mt19937_from_key(key, int(self._state[5]))
            else:
                if hasattr(seed, "squeeze"):
                    seed = seed.squeeze()
                idx = operator.index(seed)
                if idx > int(2**32 - 1) or idx < 0:
                    raise ValueError("Seed must be between 0 and 2**32 - 1")
                self._state = _native.mt19937_seed(idx)
        except TypeError:
            obj = np.asarray(seed)
            if obj.size == 0:
                raise ValueError("Seed must be non-empty")
            obj = obj.astype(np.int64, casting="safe")
            if obj.ndim != 1:
                raise ValueError("Seed array must be 1-d")
            if ((obj > int(2**32 - 1)) | (obj < 0)).any():
                raise ValueError("Seed must be between 0 and 2**32 - 1")
            obj = obj.astype(np.uint32, casting="unsafe", order="C")
            self._state = _native.mt19937_init_by_array(obj)
        self._seed_seq = None

    @property
    def state(self):
        return {
            "bit_generator": self.__class__.__name__,
            "state": {"key": self._state[6:].astype(np.uint32), "pos": int(self._state[5])},
        }

    @state.setter
    def state(self, value):
        if isinstance(value, tuple):
            if value[0] != "MT19937" or len(value) not in (3, 5):
                raise ValueError("state is not a legacy MT19937 state")
            value = {"bit_generator": "MT19937", "state": {"key": value[1], "pos": value[2]}}
        if not isinstance(value, dict):
            raise TypeError("state must be a dict")
        bitgen = value.get("bit_generator", "")
        if bitgen != self.__class__.__name__:
            raise ValueError(f"state must be for a {self.__class__.__name__} PRNG")
        key = value["state"]["key"]
        if not (isinstance(key, np.ndarray) and key.dtype == np.uint32 and key.shape == (624,)):
            key = np.array([int(key[i]) for i in range(_RK_STATE_LEN)], dtype=np.uint32)
        gauss = _native.get_gauss(self._state)
        self._state = _native.mt19937_from_key(key, operator.index(value["state"]["pos"]))
        _native.set_gauss(self._state, *gauss)


class PCG64(BitGenerator):
    """The 128-bit permuted congruential bit generator behind ``default_rng``."""

    def __init__(self, seed=None):
        BitGenerator.__init__(self, seed)
        self._state = _native.pcg64_from_seed(self._seed_seq.generate_state(4, np.uint64))

    @property
    def state(self):
        words = [int(word) for word in self._state[3:9]]
        return {
            "bit_generator": self.__class__.__name__,
            "state": {"state": words[2] * 2**64 + words[3], "inc": words[4] * 2**64 + words[5]},
            "has_uint32": words[0],
            "uinteger": words[1],
        }

    @state.setter
    def state(self, value):
        if not isinstance(value, dict):
            raise TypeError("state must be a dict")
        bitgen = value.get("bit_generator", "")
        if bitgen != self.__class__.__name__:
            raise ValueError(f"state must be for a {self.__class__.__name__} RNG")
        state = value["state"]["state"]
        inc = value["state"]["inc"]
        words = np.array(
            [state // 2**64, state % 2**64, inc // 2**64, inc % 2**64], dtype=np.uint64
        )
        self._state = _native.pcg64_from_parts(words, value["has_uint32"], value["uinteger"])

    def advance(self, delta):
        """Advance the stream as if ``delta`` draws had been made."""
        delta = int(delta) % 2**128
        _native.pcg64_advance(
            self._state, np.array([delta // 2**64, delta % 2**64], dtype=np.uint64)
        )
        return self

    def jumped(self, jumps=1):
        """A copy of the generator advanced by ``jumps`` times ``(sqrt(5) - 1) / 2 * 2**128``."""
        bit_generator = self.__class__()
        bit_generator.state = self.state
        bit_generator.advance(0x9E3779B97F4A7C15F39CC0605CEDC835 * int(jumps))
        return bit_generator


def _check_output(out, dtype, size, require_c_array):
    if out is None:
        return
    flags = out.flags
    behaved = flags.writeable and flags.aligned
    if not (behaved and (flags.c_contiguous or (flags.f_contiguous and not require_c_array))):
        req = "C-" if require_c_array else ""
        raise ValueError(
            f"Supplied output array must be {req}contiguous, writable, "
            f"aligned, and in machine byte-order."
        )
    if out.dtype != dtype:
        raise TypeError(
            "Supplied output array has the wrong type. "
            f"Expected {np.dtype(dtype)}, got {out.dtype}"
        )
    if size is not None:
        try:
            tup_size = tuple(size)
        except TypeError:
            tup_size = tuple([size])
        if tup_size != out.shape:
            raise ValueError("size must match out.shape when used together")


def _fill(kernel, state, size, out, single):
    """``double_fill`` and ``float_fill``: draws fill ``out`` in memory order."""
    if size is None and out is None:
        return float(kernel(state, 1, single)[0])
    if out is not None:
        _check_output(out, np.float32 if single else np.float64, size, False)
        target = out if out.flags.c_contiguous else out.T
        target[...] = kernel(state, target.size, single).reshape(target.shape)
        return out
    shape, count = _normalize_size(size)
    return kernel(state, count, single).reshape(shape)


def _check_non_negative(value, name):
    if not np.isnan(value) and np.signbit(value):
        raise ValueError(f"{name} < 0")


def _check_array_non_negative(values, name):
    if np.any(np.logical_and(np.logical_not(np.isnan(values)), np.signbit(values))):
        raise ValueError(f"{name} < 0")


def _broadcast_shape(size, *parameters):
    """The output shape of ``cont``'s broadcast path, checked as ``validate_output_shape``."""
    shapes = [parameter.shape for parameter in parameters]
    if size is None:
        return np.broadcast_shapes(*shapes)
    shape, _ = _normalize_size(size)
    iter_shape = np.broadcast_shapes(shape, *shapes)
    if iter_shape != shape:
        raise ValueError(
            f"Output size {shape} is not compatible with broadcast "
            f"dimensions of inputs {iter_shape}."
        )
    return shape


def _affine(draw, a, b, size, b_name, b_check):
    """``cont`` for the two-parameter distributions ``a + b * X``.

    NumPy evaluates ``a + b * X`` element by element in C doubles, drawing ``X`` in C order;
    drawing first and combining with array arithmetic gives the same values.
    """
    a_arr = np.asarray(a, dtype=np.float64)
    b_arr = np.asarray(b, dtype=np.float64)
    if a_arr.ndim == 0 and b_arr.ndim == 0:
        a_value, b_value = float(a), float(b)
        if b_check:
            _check_non_negative(b_value, b_name)
        if size is None:
            return a_value + b_value * float(draw(1)[0])
        shape, count = _normalize_size(size)
        return a_value + b_value * draw(count).reshape(shape)
    if b_check:
        _check_array_non_negative(b_arr, b_name)
    shape = _broadcast_shape(size, a_arr, b_arr)
    count = 1
    for dimension in shape:
        count *= dimension
    return a_arr + b_arr * draw(count).reshape(shape)


def _format_bounds_error(closed, low):
    # Special case low == 0 to provide a better exception for users since low = 0 is the
    # default single-argument case.
    if not np.any(low):
        comp = "<" if closed else "<="
        return f"high {comp} 0"
    comp = ">" if closed else ">="
    return f"low {comp} high"


def _integer_bounds(dtype):
    if dtype == np.dtype(np.bool_):
        return 0, 1
    info = np.iinfo(dtype)
    return int(info.min), int(info.max)


def _bounded_integers(state, low, high, size, dtype, masked, closed):
    """``_rand_{int,uint,bool}``: integers in ``[low, high)``, or ``[low, high]`` if closed."""
    name = dtype.name
    lb, ub = _integer_bounds(dtype)
    if size is not None and np.prod(size) == 0:
        return np.empty(size, dtype=dtype)
    low_arr = np.asarray(low)
    high_arr = np.asarray(high)
    if low_arr.ndim == 0 and high_arr.ndim == 0:
        low = int(low_arr)
        high = int(high_arr)
        # Subtract 1 since internal generator produces on closed interval [low, high].
        if not closed:
            high -= 1
        if low < lb:
            raise ValueError(f"low is out of bounds for {name}")
        if high > ub:
            raise ValueError(f"high is out of bounds for {name}")
        if low > high:  # -1 already subtracted, closed interval
            raise ValueError(_format_bounds_error(closed, low))
        off = np.array([low % 2**64], dtype=np.uint64)
        rng = np.array([high - low], dtype=np.uint64)
        if size is None:
            return _native.bounded_integers(state, off, rng, 1, dtype, masked)[0]
        shape, count = _normalize_size(size)
        return _native.bounded_integers(state, off, rng, count, dtype, masked).reshape(shape)
    # The broadcast path draws element by element over the broadcast bounds.
    lows = [int(value) for value in np.ravel(low_arr).tolist()]
    highs = [int(value) - (not closed) for value in np.ravel(high_arr).tolist()]
    if any(value < lb for value in lows):
        raise ValueError(f"low is out of bounds for {name}")
    if any(value > ub for value in highs):
        raise ValueError(f"high is out of bounds for {name}")
    low_arr = np.array(lows, dtype=object).reshape(low_arr.shape)
    high_arr = np.array(highs, dtype=object).reshape(high_arr.shape)
    shape = _broadcast_shape(size, low_arr, high_arr)
    lows = np.broadcast_to(low_arr, shape).ravel().tolist()
    highs = np.broadcast_to(high_arr, shape).ravel().tolist()
    if any(lo > hi for lo, hi in zip(lows, highs)):
        raise ValueError(_format_bounds_error(closed, low_arr))
    off = np.array([lo % 2**64 for lo in lows], dtype=np.uint64)
    rng = np.array([hi - lo for lo, hi in zip(lows, highs)], dtype=np.uint64)
    return _native.bounded_integers(state, off, rng, len(lows), dtype, masked).reshape(shape)


def _integer_dtype(dtype, method):
    _dtype = np.dtype(dtype)
    if _dtype.name not in _INTEGER_DTYPES:
        raise TypeError(f"Unsupported dtype {_dtype!r} for {method}")
    return _dtype


def _shuffle_order(state, n, first=1, lemire=False):
    return _native.shuffle_indices(state, n, first, lemire)


def _shuffle_sequence(state, x, axis=0):
    """``shuffle`` for arrays and mutable sequences, via the swap order of a Fisher-Yates pass."""
    if isinstance(x, np.ndarray):
        if not x.flags.writeable:
            raise ValueError("array is read-only")
        axis = normalize_axis_index(axis, np.ndim(x))
        if x.size == 0:
            # shuffling is a no-op
            return
        order = _shuffle_order(state, x.shape[axis])
        index = [slice(None)] * x.ndim
        index[axis] = order
        x[...] = x[tuple(index)]
        return
    if axis != 0:
        raise NotImplementedError("Axis argument is only supported on ndarray objects")
    n = len(x)
    order = _shuffle_order(state, n).tolist()
    items = [x[k] for k in order]
    for i in range(n):
        x[i] = items[i]


class Generator:
    """NumPy's recommended random interface, drawing from a bit generator."""

    def __init__(self, bit_generator):
        if not isinstance(bit_generator, BitGenerator):
            raise AttributeError(
                f"'{type(bit_generator).__name__}' object has no attribute 'capsule'"
            )
        self._bit_generator = bit_generator

    def __repr__(self):
        return f"{self} at 0x{id(self):X}"

    def __str__(self):
        return f"{type(self).__name__}({type(self._bit_generator).__name__})"

    @property
    def bit_generator(self):
        return self._bit_generator

    def spawn(self, n_children):
        """Create new independent child generators."""
        return [type(self)(g) for g in self._bit_generator.spawn(n_children)]

    def random(self, size=None, dtype=np.float64, out=None):
        """Return random floats in the half-open interval [0.0, 1.0)."""
        _dtype = np.dtype(dtype)
        if _dtype == np.float64:
            single = False
        elif _dtype == np.float32:
            single = True
        else:
            raise TypeError(f"Unsupported dtype {_dtype!r} for random")
        return _fill(_native.standard_uniform, self._bit_generator._state, size, out, single)

    def integers(self, low, high=None, size=None, dtype=np.int64, endpoint=False):
        """Return random integers from ``low`` (inclusive) to ``high`` (exclusive)."""
        if high is None:
            high = low
            low = 0
        _dtype = _integer_dtype(dtype, "integers")
        ret = _bounded_integers(
            self._bit_generator._state, low, high, size, _dtype, False, endpoint
        )
        if size is None and (dtype is bool or dtype is int):
            if np.array(ret).shape == ():
                return dtype(ret)
        return ret

    def uniform(self, low=0.0, high=1.0, size=None):
        """Draw samples from a uniform distribution over ``[low, high)``."""
        alow = np.asarray(low, dtype=np.float64)
        ahigh = np.asarray(high, dtype=np.float64)
        draw = self._uniform_draws
        if alow.ndim == ahigh.ndim == 0:
            rng = float(high) - float(low)
            if not np.isfinite(rng):
                raise OverflowError("high - low range exceeds valid bounds")
            return _affine(draw, float(low), rng, size, "high - low", True)
        arange = np.subtract(ahigh, alow)
        if not np.all(np.isfinite(arange)):
            raise OverflowError("Range exceeds valid bounds")
        return _affine(draw, alow, arange, size, "high - low", True)

    def _uniform_draws(self, count):
        return _native.standard_uniform(self._bit_generator._state, count, False)

    def _normal_draws(self, count):
        return _native.standard_normal(self._bit_generator._state, count, False)

    def standard_normal(self, size=None, dtype=np.float64, out=None):
        """Draw samples from a standard Normal distribution (mean=0, stdev=1)."""
        _dtype = np.dtype(dtype)
        if _dtype == np.float64:
            single = False
        elif _dtype == np.float32:
            single = True
        else:
            raise TypeError(f"Unsupported dtype {_dtype!r} for standard_normal")
        return _fill(_native.standard_normal, self._bit_generator._state, size, out, single)

    def normal(self, loc=0.0, scale=1.0, size=None):
        """Draw random samples from a normal (Gaussian) distribution."""
        return _affine(self._normal_draws, loc, scale, size, "scale", True)

    def choice(self, a, size=None, replace=True, p=None, axis=0, shuffle=True):
        """Generates a random sample from a given array."""
        state = self._bit_generator._state
        a_original = a
        a = np.asarray(a)
        if a.ndim == 0:
            try:
                # __index__ must return an integer by python rules.
                pop_size = operator.index(a.item())
            except TypeError as exc:
                raise ValueError(
                    f"a must be a sequence or an integer, not {type(a_original)}"
                ) from exc
            if pop_size <= 0 and np.prod(size) != 0:
                raise ValueError("a must be a positive integer unless no samples are taken")
        else:
            pop_size = a.shape[axis]
            if pop_size == 0 and np.prod(size) != 0:
                raise ValueError("a cannot be empty unless no samples are taken")

        if p is not None:
            d = len(p)
            atol = np.sqrt(np.finfo(np.float64).eps)
            if isinstance(p, np.ndarray):
                if np.issubdtype(p.dtype, np.floating):
                    atol = max(atol, np.sqrt(np.finfo(p.dtype).eps))
            p = np.array(p, dtype=np.float64)
            if p.ndim != 1:
                raise ValueError("p must be 1-dimensional")
            if p.size != pop_size:
                raise ValueError("a and p must have same size")
            p_sum = _kahan_sum(p.tolist()[:d])
            if np.isnan(p_sum):
                raise ValueError("Probabilities contain NaN")
            if np.logical_or.reduce(p < 0):
                raise ValueError("Probabilities are not non-negative")
            if abs(p_sum - 1.0) > atol:
                raise ValueError(
                    "Probabilities do not sum to 1. See Notes section of docstring for more "
                    "information."
                )

        # `shape == None` means `shape == ()`, but with scalar unpacking at the end.
        is_scalar = size is None
        if not is_scalar:
            shape = size
            size = np.prod(shape, dtype=np.intp)
        else:
            shape = ()
            size = 1

        if replace:
            if p is not None:
                cdf = p.cumsum()
                cdf /= cdf[-1]
                uniform_samples = self.random(shape)
                idx = cdf.searchsorted(uniform_samples, side="right")
                # searchsorted returns a scalar
                idx = np.asarray(idx, dtype=np.int64)
            else:
                idx = self.integers(0, pop_size, size=shape, dtype=np.int64)
        else:
            if size > pop_size:
                raise ValueError(
                    "Cannot take a larger sample than population when replace is False"
                )
            elif size < 0:
                raise ValueError("negative dimensions are not allowed")

            if p is not None:
                if np.count_nonzero(p > 0) < size:
                    raise ValueError("Fewer non-zero entries in p than size")
                n_uniq = 0
                p = p.copy()
                found = np.zeros(shape, dtype=np.int64)
                flat_found = found.ravel()
                while n_uniq < size:
                    x = self.random((size - n_uniq,))
                    if n_uniq > 0:
                        p[flat_found[0:n_uniq]] = 0
                    cdf = np.cumsum(p)
                    cdf /= cdf[-1]
                    new = cdf.searchsorted(x, side="right")
                    _, unique_indices = np.unique(new, return_index=True)
                    unique_indices.sort()
                    new = new.take(unique_indices)
                    flat_found[n_uniq : n_uniq + new.size] = new
                    n_uniq += new.size
                idx = found
            else:
                size_i = int(size)
                pop_size_i = int(pop_size)
                # This is a heuristic tuning. should be improvable
                cutoff = 50 if shuffle else 20
                if pop_size_i > 10000 and (size_i > (pop_size_i // cutoff)):
                    # Tail shuffle size elements
                    order = _shuffle_order(
                        state, pop_size_i, max(pop_size_i - size_i, 1), lemire=True
                    )
                    idx = order[(pop_size_i - size_i) :].copy()
                else:
                    # Floyd's algorithm
                    idx = _native.floyd_sample(state, pop_size_i, size_i, bool(shuffle))
                idx = idx.reshape(shape)

        if is_scalar and isinstance(idx, np.ndarray):
            # In most cases a scalar will have been made an array
            idx = idx.item(0)

        # Use samples as indices for a if a is array-like
        if a.ndim == 0:
            return idx

        if not is_scalar and idx.ndim == 0 and a.ndim == 1:
            # If size == () then the user requested a 0-d array as opposed to a scalar object
            # when size is None.
            res = np.empty((), dtype=a.dtype)
            res[()] = a[idx]
            return res

        return a.take(np.asarray(idx, dtype=np.intp), axis=axis)

    def shuffle(self, x, axis=0):
        """Modify an array or sequence in-place by shuffling its contents."""
        _shuffle_sequence(self._bit_generator._state, x, axis)

    def permutation(self, x, axis=0):
        """Randomly permute a sequence, or return a permuted range."""
        if isinstance(x, (int, np.integer)):
            arr = np.arange(x)
            self.shuffle(arr)
            return arr

        arr = np.asarray(x)
        axis = normalize_axis_index(axis, arr.ndim)

        # shuffle has fast-path for 1-d
        if arr.ndim == 1:
            # Return a copy if same memory
            if np.may_share_memory(arr, x):
                arr = np.array(arr)
            self.shuffle(arr)
            return arr

        # Shuffle index array, dtype to ensure fast path
        idx = np.arange(arr.shape[axis], dtype=np.intp)
        self.shuffle(idx)
        slices = [slice(None)] * arr.ndim
        slices[axis] = idx
        return arr[tuple(slices)]


def _kahan_sum(values):
    """``kahan_sum`` from ``_common.pyx``, which ``choice`` uses to validate ``p``."""
    if len(values) <= 0:
        return 0.0
    total = values[0]
    c = 0.0
    for value in values[1:]:
        y = value - c
        t = total + y
        c = (t - total) - y
        total = t
    return total


def default_rng(seed=None):
    """Construct a new Generator with the default BitGenerator (PCG64)."""
    if isinstance(seed, BitGenerator):
        # We were passed a BitGenerator, so just wrap it up.
        return Generator(seed)
    elif isinstance(seed, Generator):
        # Pass through a Generator.
        return seed
    elif isinstance(seed, RandomState):
        return Generator(seed._bit_generator)
    # Otherwise we need to instantiate a new BitGenerator and Generator as normal.
    return Generator(PCG64(seed))


class RandomState:
    """NumPy's legacy random interface on the Mersenne Twister."""

    def __init__(self, seed=None):
        if seed is None:
            bit_generator = MT19937()
        elif not isinstance(seed, BitGenerator):
            bit_generator = MT19937()
            bit_generator._legacy_seeding(seed)
        else:
            bit_generator = seed
        self._initialize_bit_generator(bit_generator)

    def __repr__(self):
        return f"{self} at 0x{id(self):X}"

    def __str__(self):
        return f"{self.__class__.__name__}({self._bit_generator.__class__.__name__})"

    def _initialize_bit_generator(self, bit_generator):
        self._bit_generator = bit_generator
        self._reset_gauss()

    def _reset_gauss(self):
        _native.set_gauss(self._bit_generator._state, 0, 0.0)

    def seed(self, seed=None):
        """Reseed the legacy MT19937 stream."""
        if not isinstance(self._bit_generator, MT19937):
            raise TypeError("can only re-seed a MT19937 BitGenerator")
        self._bit_generator._legacy_seeding(seed)
        self._reset_gauss()

    def get_state(self, legacy=True):
        """Return a tuple (or dict) representing the internal state of the generator."""
        st = self._bit_generator.state
        st["has_gauss"], st["gauss"] = _native.get_gauss(self._bit_generator._state)
        if st["bit_generator"] != "MT19937" and legacy:
            import warnings

            warnings.warn(
                "get_state and legacy can only be used with the MT19937 BitGenerator. To "
                "silence this warning, set `legacy` to False.",
                RuntimeWarning,
                stacklevel=2,
            )
            legacy = False
        if legacy:
            return (
                st["bit_generator"],
                st["state"]["key"],
                st["state"]["pos"],
                st["has_gauss"],
                st["gauss"],
            )
        return st

    def set_state(self, state):
        """Set the internal state of the generator from a tuple or dict."""
        if isinstance(state, dict):
            if "bit_generator" not in state or "state" not in state:
                raise ValueError("state dictionary is not valid.")
            st = state
        else:
            if not isinstance(state, (tuple, list)):
                raise TypeError("state must be a dict or a tuple.")
            if state[0] != "MT19937":
                raise ValueError("set_state can only be used with legacy MT19937 state instances.")
            st = {"bit_generator": state[0], "state": {"key": state[1], "pos": state[2]}}
            if len(state) > 3:
                st["has_gauss"] = state[3]
                st["gauss"] = state[4]
        gauss = st.get("gauss", 0.0)
        has_gauss = st.get("has_gauss", 0)
        self._bit_generator.state = st
        _native.set_gauss(self._bit_generator._state, has_gauss, gauss)

    def random_sample(self, size=None):
        """Return random floats in the half-open interval [0.0, 1.0)."""
        return _fill(_native.standard_uniform, self._bit_generator._state, size, None, False)

    def random(self, size=None):
        """Return random floats in the half-open interval [0.0, 1.0)."""
        return self.random_sample(size=size)

    def rand(self, *args):
        """Random values in a given shape."""
        if len(args) == 0:
            return self.random_sample()
        return self.random_sample(size=args)

    def randn(self, *args):
        """Return samples from the "standard normal" distribution."""
        if len(args) == 0:
            return self.standard_normal()
        return self.standard_normal(size=args)

    def _gauss_draws(self, count):
        return _native.legacy_gauss(self._bit_generator._state, count)

    def standard_normal(self, size=None):
        """Draw samples from a standard Normal distribution (mean=0, stdev=1)."""
        if size is None:
            return float(self._gauss_draws(1)[0])
        shape, count = _normalize_size(size)
        return self._gauss_draws(count).reshape(shape)

    def normal(self, loc=0.0, scale=1.0, size=None):
        """Draw random samples from a normal (Gaussian) distribution."""
        return _affine(self._gauss_draws, loc, scale, size, "scale", True)

    def uniform(self, low=0.0, high=1.0, size=None):
        """Draw samples from a uniform distribution over ``[low, high)``."""
        alow = np.asarray(low, dtype=np.float64)
        ahigh = np.asarray(high, dtype=np.float64)
        draw = self._uniform_draws
        if alow.ndim == ahigh.ndim == 0:
            rng = float(high) - float(low)
            if not np.isfinite(rng):
                raise OverflowError("Range exceeds valid bounds")
            return _affine(draw, float(low), rng, size, "", False)
        arange = np.subtract(ahigh, alow)
        if not np.all(np.isfinite(arange)):
            raise OverflowError("Range exceeds valid bounds")
        return _affine(draw, alow, arange, size, "", False)

    def _uniform_draws(self, count):
        return _native.standard_uniform(self._bit_generator._state, count, False)

    def randint(self, low, high=None, size=None, dtype=int):
        """Return random integers from ``low`` (inclusive) to ``high`` (exclusive)."""
        if high is None:
            high = low
            low = 0
        _dtype = _integer_dtype(dtype if dtype is not int else "long", "randint")
        # The legacy stream always uses masked rejection, as NumPy keeps it for compatibility.
        ret = _bounded_integers(self._bit_generator._state, low, high, size, _dtype, True, False)
        if size is None and (dtype is bool or dtype is int):
            if np.array(ret).shape == ():
                return dtype(ret)
        return ret

    def random_integers(self, low, high=None, size=None):
        """Random integers between ``low`` and ``high``, inclusive (deprecated)."""
        import warnings

        # NumPy warns from Cython, which has no frame, so `stacklevel=2` names the same caller.
        if high is None:
            warnings.warn(
                f"This function is deprecated. Please call randint(1, {low} + 1) instead",
                DeprecationWarning,
                stacklevel=2,
            )
            high = low
            low = 1
        else:
            warnings.warn(
                f"This function is deprecated. Please call randint({low}, {high} + 1) instead",
                DeprecationWarning,
                stacklevel=2,
            )
        return self.randint(low, int(high) + 1, size=size, dtype="l")

    def choice(self, a, size=None, replace=True, p=None):
        """Generates a random sample from a given 1-D array."""
        a = np.asarray(a)
        if a.ndim == 0:
            try:
                # __index__ must return an integer by python rules.
                pop_size = operator.index(a.item())
            except TypeError:
                raise ValueError("a must be 1-dimensional or an integer")
            if pop_size <= 0 and np.prod(size) != 0:
                raise ValueError("a must be greater than 0 unless no samples are taken")
        elif a.ndim != 1:
            raise ValueError("a must be 1-dimensional")
        else:
            pop_size = a.shape[0]
            if pop_size == 0 and np.prod(size) != 0:
                raise ValueError("'a' cannot be empty unless no samples are taken")

        if p is not None:
            d = len(p)
            atol = np.sqrt(np.finfo(np.float64).eps)
            if isinstance(p, np.ndarray):
                if np.issubdtype(p.dtype, np.floating):
                    atol = max(atol, np.sqrt(np.finfo(p.dtype).eps))
            p = np.array(p, dtype=np.float64)
            if p.ndim != 1:
                raise ValueError("'p' must be 1-dimensional")
            if p.size != pop_size:
                raise ValueError("'a' and 'p' must have same size")
            p_sum = _kahan_sum(p.tolist()[:d])
            if np.isnan(p_sum):
                raise ValueError("probabilities contain NaN")
            if np.logical_or.reduce(p < 0):
                raise ValueError("probabilities are not non-negative")
            if abs(p_sum - 1.0) > atol:
                raise ValueError("probabilities do not sum to 1")

        # `shape == None` means `shape == ()`, but with scalar unpacking at the end.
        is_scalar = size is None
        if not is_scalar:
            shape = size
            size = np.prod(shape, dtype=np.intp)
        else:
            shape = ()
            size = 1

        if replace:
            if p is not None:
                cdf = p.cumsum()
                cdf /= cdf[-1]
                uniform_samples = self.random_sample(shape)
                idx = cdf.searchsorted(uniform_samples, side="right")
                # searchsorted returns a scalar
                idx = np.asarray(idx).astype(np.long, casting="unsafe")
            else:
                idx = self.randint(0, pop_size, size=shape)
        else:
            if size > pop_size:
                raise ValueError(
                    "Cannot take a larger sample than population when 'replace=False'"
                )
            elif size < 0:
                raise ValueError("Negative dimensions are not allowed")

            if p is not None:
                if np.count_nonzero(p > 0) < size:
                    raise ValueError("Fewer non-zero entries in p than size")
                n_uniq = 0
                p = p.copy()
                found = np.zeros(shape, dtype=np.long)
                flat_found = found.ravel()
                while n_uniq < size:
                    x = self.rand(size - n_uniq)
                    if n_uniq > 0:
                        p[flat_found[0:n_uniq]] = 0
                    cdf = np.cumsum(p)
                    cdf /= cdf[-1]
                    new = cdf.searchsorted(x, side="right")
                    _, unique_indices = np.unique(new, return_index=True)
                    unique_indices.sort()
                    new = new.take(unique_indices)
                    flat_found[n_uniq : n_uniq + new.size] = new
                    n_uniq += new.size
                idx = found
            else:
                idx = self.permutation(pop_size)[:size]
                idx = idx.reshape(shape)

        if is_scalar and isinstance(idx, np.ndarray):
            # In most cases a scalar will have been made an array
            idx = idx.item(0)

        # Use samples as indices for a if a is array-like
        if a.ndim == 0:
            return idx

        if not is_scalar and idx.ndim == 0:
            # If size == () then the user requested a 0-d array as opposed to a scalar object
            # when size is None.
            res = np.empty((), dtype=a.dtype)
            res[()] = a[idx]
            return res

        return a[idx]

    def shuffle(self, x):
        """Modify a sequence in-place by shuffling its contents."""
        _shuffle_sequence(self._bit_generator._state, x)

    def permutation(self, x):
        """Randomly permute a sequence, or return a permuted range."""
        if isinstance(x, (int, np.integer)):
            # keep using long as the default here (main numpy switched to intp)
            arr = np.arange(x, dtype=np.result_type(x, np.long))
            self.shuffle(arr)
            return arr

        arr = np.asarray(x)
        if arr.ndim < 1:
            raise IndexError("x must be an integer or at least 1-dimensional")

        # shuffle has fast-path for 1-d
        if arr.ndim == 1:
            # Return a copy if same memory
            if np.may_share_memory(arr, x):
                arr = np.array(arr)
            self.shuffle(arr)
            return arr

        # Shuffle index array, dtype to ensure fast path
        idx = np.arange(arr.shape[0], dtype=np.intp)
        self.shuffle(idx)
        return arr[idx]


_rand = RandomState()

choice = _rand.choice
get_state = _rand.get_state
normal = _rand.normal
permutation = _rand.permutation
rand = _rand.rand
randint = _rand.randint
randn = _rand.randn
random = _rand.random
random_integers = _rand.random_integers
random_sample = _rand.random_sample
set_state = _rand.set_state
shuffle = _rand.shuffle
standard_normal = _rand.standard_normal
uniform = _rand.uniform


def seed(seed=None):
    """Reseed the global legacy stream."""
    if isinstance(_rand._bit_generator, MT19937):
        return _rand.seed(seed)
    else:
        bg_type = type(_rand._bit_generator)
        _rand.set_state(bg_type(seed).state)


def get_bit_generator():
    """Return the bit generator behind the global legacy stream."""
    return _rand._bit_generator


def set_bit_generator(bitgen):
    """Replace the bit generator behind the global legacy stream."""
    _rand._initialize_bit_generator(bitgen)


# Old aliases that should not be removed
def sample(*args, **kwargs):
    return _rand.random_sample(*args, **kwargs)


def ranf(*args, **kwargs):
    return _rand.random_sample(*args, **kwargs)
