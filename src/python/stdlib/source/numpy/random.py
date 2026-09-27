"""numpy.random: pseudo-random number generation.

Two generator families are implemented:

- ``Generator`` (returned by ``default_rng``), backed by a bit generator
  (``PCG64`` by default, or ``MT19937``). Bounded integers use Lemire's
  method; the normal/exponential ziggurat is this module's own clean-room
  implementation of the published Marsaglia-Tsang algorithm and does not
  reproduce NumPy's exact stream (see docs/numpy.md). ``standard_gamma`` and
  the distributions built from it inherit that difference whenever they
  draw a normal deviate.
- ``RandomState`` (legacy), always backed by ``MT19937``, matching NumPy's
  original generator bit for bit: ``init_genrand``/``init_by_array``
  seeding, masked-rejection bounded integers, and the Marsaglia polar-method
  Gaussian (with its one-value cache).

The per-element draw loops live in the native ``_numpy_random`` module. This
file owns object state, argument defaults, dtype/shape resolution, and
parameter broadcasting (via ``numpy.broadcast_to``), matching NumPy's own
split between its object layer and its generation loops. A bit generator's
state is threaded through every native call as a small Python list
(``[kind, *fields]``) and reassigned on return, so state changes are always
explicit and never hidden in native-side mutation.
"""

import numpy as _np
import _numpy_random as _nr

# ---------------------------------------------------------------------------
# SeedSequence: NumPy's entropy-mixing seed expander.
#
# This is Melissa O'Neill's public "seed_seq_fe" design
# (https://www.pcg-random.org/posts/developing-a-seed_seq-alternative.html),
# which NumPy documents using the same hashmix/mix building blocks and pool
# size. The constants and control flow below were derived from that public
# description and confirmed against NumPy 2.5.3's own generate_state/spawn
# output treated strictly as a black box (matching scalar, multi-word, and
# multi-level-spawn entropy, and the low-word-first uint64 packing).
# ---------------------------------------------------------------------------

_MASK32 = 0xFFFFFFFF
_INIT_A = 0x43B0D7E5
_MULT_A = 0x931E8875
_INIT_B = 0x8B51F9DD
_MULT_B = 0x58F38DED
_MIX_MULT_L = 0xCA01F9DD
_MIX_MULT_R = 0x4973F715
_XSHIFT = 16
_DEFAULT_POOL_SIZE = 4


def _hashmix(value, hash_const):
    value = (value ^ hash_const[0]) & _MASK32
    hash_const[0] = (hash_const[0] * _MULT_A) & _MASK32
    value = (value * hash_const[0]) & _MASK32
    value ^= value >> _XSHIFT
    return value & _MASK32


def _mix(x, y):
    result = (_MIX_MULT_L * x - _MIX_MULT_R * y) & _MASK32
    result ^= result >> _XSHIFT
    return result & _MASK32


def _is_plain_int(value):
    return isinstance(value, int) and not isinstance(value, bool)


def _int_to_u32_words(value):
    if value < 0:
        raise ValueError("expected a non-negative integer")
    if value == 0:
        return [0]
    words = []
    while value > 0:
        words.append(value & _MASK32)
        value >>= 32
    return words


def _coerce_entropy_words(entropy):
    if _is_plain_int(entropy):
        return _int_to_u32_words(entropy)
    try:
        items = list(entropy)
    except TypeError:
        raise TypeError("SeedSequence expects int or sequence of ints") from None
    words = []
    for item in items:
        if not _is_plain_int(item):
            raise TypeError("SeedSequence expects int or sequence of ints")
        words.extend(_int_to_u32_words(item))
    return words


def _mix_entropy(pool_size, entropy):
    pool = [0] * pool_size
    hash_const = [_INIT_A]
    for i in range(pool_size):
        pool[i] = _hashmix(entropy[i] if i < len(entropy) else 0, hash_const)
    # Mix all pool words together so every input bit affects every output word.
    for i_src in range(pool_size):
        for i_dst in range(pool_size):
            if i_src != i_dst:
                pool[i_dst] = _mix(pool[i_dst], _hashmix(pool[i_src], hash_const))
    # Mix in any entropy words past the pool size, each against every slot.
    for i_src in range(pool_size, len(entropy)):
        for i_dst in range(pool_size):
            pool[i_dst] = _mix(pool[i_dst], _hashmix(entropy[i_src], hash_const))
    return pool


def _generate_words(pool, n_words):
    hash_const = _INIT_B
    out = []
    src = 0
    for _ in range(n_words):
        value = pool[src] ^ hash_const
        hash_const = (hash_const * _MULT_B) & _MASK32
        value = (value * hash_const) & _MASK32
        value ^= value >> _XSHIFT
        out.append(value & _MASK32)
        src += 1
        if src == len(pool):
            src = 0
    return out


class SeedSequence:
    """Spread entropy across a small pool and expand it to any number of words.

    ``entropy`` is an int, a sequence of ints, or ``None``. Shellsim grants
    no host entropy source (see ``random.py``'s module docstring for the
    same rule applied to the stdlib ``random`` module), so ``None`` maps to
    a fixed value rather than OS randomness; give an explicit seed for a
    reproducible stream, which is the only kind shellsim can produce anyway.
    """

    def __init__(self, entropy=None, *, spawn_key=(), pool_size=_DEFAULT_POOL_SIZE):
        if entropy is None:
            entropy = 0
        if not (_is_plain_int(entropy) or hasattr(entropy, "__iter__")):
            raise TypeError("SeedSequence expects int or sequence of ints")
        self.entropy = entropy
        self.spawn_key = tuple(spawn_key)
        self.pool_size = pool_size
        self.n_children_spawned = 0
        run_words = _coerce_entropy_words(entropy)
        spawn_words = []
        for item in self.spawn_key:
            spawn_words.extend(_int_to_u32_words(item))
        if spawn_words and len(run_words) < pool_size:
            run_words = run_words + [0] * (pool_size - len(run_words))
        self._pool = _mix_entropy(pool_size, run_words + spawn_words)

    def generate_state(self, n_words, dtype=_np.uint32):
        dtype_name = _np.dtype(dtype).name
        if dtype_name == "uint32":
            return _np.array(_generate_words(self._pool, n_words), dtype=_np.uint32)
        if dtype_name == "uint64":
            words = _generate_words(self._pool, n_words * 2)
            combined = [words[2 * i] | (words[2 * i + 1] << 32) for i in range(n_words)]
            return _np.array(combined, dtype=_np.uint64)
        raise ValueError("SeedSequence.generate_state only support uint32 or uint64")

    def spawn(self, n_children):
        children = []
        for _ in range(n_children):
            children.append(
                SeedSequence(
                    self.entropy,
                    spawn_key=self.spawn_key + (self.n_children_spawned,),
                    pool_size=self.pool_size,
                )
            )
            self.n_children_spawned += 1
        return children


# ---------------------------------------------------------------------------
# Bit generators.
# ---------------------------------------------------------------------------


def _mt_state_from_seed_sequence(seed_sequence):
    key = seed_sequence.generate_state(624).astype(_np.uint32)
    key[0] = 0x80000000
    return ["MT19937", key, 623]


class MT19937:
    """The Mersenne Twister (Matsumoto & Nishimura 1998) bit generator.

    A scalar or array-like seed is expanded through `SeedSequence` first
    (``key = SeedSequence(seed).generate_state(624)`` with ``key[0]``
    overwritten to ``0x80000000`` and the twist position left at 623, not
    624, so the constructor's own state is one word into the array rather
    than needing an immediate re-twist). This backs `Generator` by default
    only when `PCG64` isn't used; legacy `RandomState` seeds its own
    MT19937 directly through `init_genrand`/`init_by_array` instead (see
    `RandomState.seed`), bypassing `SeedSequence` entirely for exact
    backward compatibility with NumPy's original generator.
    """

    def __init__(self, seed=None):
        self.seed_seq = seed if isinstance(seed, SeedSequence) else SeedSequence(seed)
        self._state = _mt_state_from_seed_sequence(self.seed_seq)

    @property
    def state(self):
        return {
            "bit_generator": "MT19937",
            "state": {"key": self._state[1].copy(), "pos": self._state[2]},
        }

    @state.setter
    def state(self, value):
        key = _np.array(value["state"]["key"], dtype=_np.uint32)
        self._state = ["MT19937", key, int(value["state"]["pos"])]

    def random_raw(self, size=None):
        n = 1 if size is None else int(size)
        values, new_state = _nr._raw_fill(self._state, n)
        self._state = new_state
        return int(values[0]) if size is None else values


def _pcg_state_from_seed_sequence(seed_sequence):
    words = [int(word) for word in seed_sequence.generate_state(4, _np.uint64)]
    return _nr._pcg_seed(words[0], words[1], words[2], words[3])


class PCG64:
    """PCG64 (O'Neill 2014), XSL-RR variant: the default `Generator` bit generator.

    128-bit LCG state advanced by PCG's published 128-bit multiplier,
    output through the xorshift-low/random-rotate (XSL RR) function that
    folds state to 64 bits. `advance` and `jumped` are both the same
    generic LCG jump-ahead by repeated doubling; `jumped`'s distance-per-
    jump ordinarily comes from PCG's C++ template parameters, which NumPy
    does not expose through Python, so its coefficients were recovered by
    treating ``jumped`` as an unknown affine map over the 128-bit state and
    solving for it from `PCG64(1).jumped(3)`'s published output (see
    `random/bitgen.rs`).
    """

    def __init__(self, seed=None):
        self.seed_seq = seed if isinstance(seed, SeedSequence) else SeedSequence(seed)
        self._state = _pcg_state_from_seed_sequence(self.seed_seq)

    @property
    def state(self):
        return {
            "bit_generator": "PCG64",
            "state": {"state": int(self._state[1]), "inc": int(self._state[2])},
            "has_uint32": int(self._state[3]),
            "uinteger": int(self._state[4]),
        }

    @state.setter
    def state(self, value):
        self._state = [
            "PCG64",
            int(value["state"]["state"]),
            int(value["state"]["inc"]),
            bool(value["has_uint32"]),
            int(value["uinteger"]),
        ]

    def advance(self, delta):
        self._state = _nr._pcg_advance(self._state, int(delta))
        return self

    def jumped(self, iterations=1):
        self._state = _nr._pcg_jumped(self._state, int(iterations))
        return self

    def random_raw(self, size=None):
        n = 1 if size is None else int(size)
        values, new_state = _nr._raw_fill(self._state, n)
        self._state = new_state
        return int(values[0]) if size is None else values


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

    Mirrors NumPy's rule: with no explicit ``size``, the output shape is
    the parameters' own broadcast shape; with an explicit ``size``, the
    parameters must broadcast to it or the call fails with "shape
    mismatch" (NumPy's own wording for this case).
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
    cumulative = p_arr.astype(float).tolist()
    running = 0.0
    for i, weight in enumerate(cumulative):
        running += weight
        cumulative[i] = running
    cumulative[-1] = 1.0
    base, new_state = _nr._uniform01_fill(raw_state(), [n_samples], "float64")
    store_state(new_state)
    result = []
    for value in base.tolist():
        pick = len(cumulative) - 1
        for i, threshold in enumerate(cumulative):
            if value < threshold:
                pick = i
                break
        result.append(pick)
    return _np.array(result, dtype=_np.int64)


def _weighted_choice_without_replacement(raw_state, store_state, p_arr, n_samples):
    remaining_p = p_arr.astype(float).tolist()
    remaining_idx = list(range(len(remaining_p)))
    result = []
    for _ in range(n_samples):
        total = sum(remaining_p)
        base, new_state = _nr._uniform01_fill(raw_state(), [1], "float64")
        store_state(new_state)
        target = float(base[0]) * total
        cumulative = 0.0
        pick = len(remaining_p) - 1
        for i, weight in enumerate(remaining_p):
            cumulative += weight
            if target < cumulative:
                pick = i
                break
        result.append(remaining_idx.pop(pick))
        remaining_p.pop(pick)
    return _np.array(result, dtype=_np.int64)


def _choice_indices(raw_state, store_state, pop_size, shape, replace, p, legacy):
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
            if legacy:
                full, new_state = _nr._shuffle_indices(raw_state(), pop_size)
            else:
                full, new_state = _nr._sample_without_replacement(
                    raw_state(), pop_size, n_samples
                )
            store_state(new_state)
            idx = _np.asarray(full, dtype=_np.int64)[:n_samples]
        else:
            idx = _weighted_choice_without_replacement(raw_state, store_state, p_arr, n_samples)
    else:
        if p_arr is None:
            low_b = _np.zeros(n_samples, dtype=_np.int64)
            count_b = _np.full(n_samples, float(pop_size), dtype=_np.float64)
            idx, new_state = _nr._bounded_int_fill(raw_state(), low_b, count_b, legacy, "int64")
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
    """NumPy-style random number generator over any bit generator.

    Uniform draws come straight from the bit generator's words (Lemire's
    method for bounded integers; the top 53 bits of a 64-bit draw, or two
    tempered MT19937 words, for doubles in ``[0, 1)``); those match NumPy
    2.5.3 bit for bit for both `PCG64` and `MT19937`. The normal and
    exponential ziggurat is this module's own implementation and does not
    reproduce NumPy's exact stream (see docs/numpy.md); `standard_gamma`
    (and `gamma`, `chisquare`, `f`, `standard_t`, which are built on it)
    inherit that difference whenever they consume a normal deviate.
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
        count = high_incl.astype(_np.float64) - low_b.astype(_np.float64) + 1.0
        values, new_state = _nr._bounded_int_fill(
            self._raw_state(), low_b, count, False, dtype_name
        )
        self._store_state(new_state)
        result = values if dtype_name == "int64" else values.astype(dtype_name)
        return _scalar(result) if target == () else result

    def choice(self, a, size=None, replace=True, p=None):
        pop_size, values_source = _choice_population(a)
        shape = _shape_of(size)
        idx = _choice_indices(
            self._raw_state, self._store_state, pop_size, shape, replace, p, False
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
        values, new_state = _nr._standard_exponential_fill(
            self._raw_state(), shape, dtype_name, method
        )
        self._store_state(new_state)
        if out is not None:
            out[...] = values
            return out
        return float(_scalar(values)) if shape == () else values

    def exponential(self, scale=1.0, size=None):
        shape, (scale_a,) = _broadcast_params(size, (scale,))
        if _np.any(scale_a < 0):
            raise ValueError("scale < 0")
        base, new_state = _nr._standard_exponential_fill(
            self._raw_state(), shape, "float64", "zig"
        )
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
    """Return a new `Generator` seeded from `seed` through `SeedSequence`,
    backed by `PCG64` (NumPy's default bit generator since 1.17).
    """
    if isinstance(seed, (Generator,)):
        return seed
    if isinstance(seed, (MT19937, PCG64)):
        return Generator(seed)
    seq = seed if isinstance(seed, SeedSequence) else SeedSequence(seed)
    return Generator(PCG64(seq))


# ---------------------------------------------------------------------------
# RandomState (legacy)
# ---------------------------------------------------------------------------


def _mt_seed_scalar(value):
    if not 0 <= value <= 2**32 - 1:
        raise ValueError("Seed must be between 0 and 2**32 - 1")
    return _nr._mt_seed_genrand(value)


def _mt_seed_array(words):
    for word in words:
        if not 0 <= word <= 2**32 - 1:
            raise ValueError("Seed must be between 0 and 2**32 - 1")
    return _nr._mt_seed_array(_np.array(words, dtype=_np.int64))


class RandomState:
    """Legacy MT19937-backed generator, bit-for-bit compatible with NumPy's
    original `RandomState`: `init_genrand`/`init_by_array` seeding, masked
    rejection for bounded integers, and the Marsaglia polar-method Gaussian
    (with its one-value cache, threaded through `get_state`/`set_state`).
    """

    def __init__(self, seed=None):
        self.seed(seed)

    def seed(self, seed=None):
        if seed is None:
            self._state = _mt_seed_scalar(0)
        elif isinstance(seed, (int, _np.integer)) and not isinstance(seed, bool):
            self._state = _mt_seed_scalar(int(seed))
        else:
            try:
                words = [int(value) for value in seed]
            except TypeError:
                raise TypeError("Seed must be None, an int, or an array of ints") from None
            self._state = _mt_seed_array(words)
        self._has_gauss = False
        self._cached_gauss = 0.0

    def _raw_state(self):
        return [self._state[0], self._state[1], self._state[2], self._has_gauss, self._cached_gauss]

    def _store_state(self, new_state):
        self._state = [new_state[0], new_state[1], new_state[2]]
        self._has_gauss = bool(new_state[3])
        self._cached_gauss = float(new_state[4])

    def get_state(self, legacy=True):
        return ("MT19937", self._state[1].copy(), self._state[2], int(self._has_gauss), self._cached_gauss)

    def set_state(self, state):
        _, key, pos, has_gauss, cached = state
        self._state = ["MT19937", _np.array(key, dtype=_np.uint32), int(pos)]
        self._has_gauss = bool(has_gauss)
        self._cached_gauss = float(cached)

    def rand(self, *shape):
        return self.random_sample(shape if shape else None)

    def randn(self, *shape):
        return self.standard_normal(shape if shape else None)

    def random_sample(self, size=None):
        shape = _shape_of(size)
        values, new_state = _nr._uniform01_fill(self._raw_state(), shape, "float64")
        self._store_state(new_state)
        return float(_scalar(values)) if shape == () else values

    random = random_sample

    def randint(self, low, high=None, size=None, dtype=_np.int64):
        dtype_name = _dtype_name(dtype)
        if dtype_name not in _INT_BOUNDS:
            raise TypeError(f"Unsupported dtype {dtype_name} for randint")
        dmin, dmax = _INT_BOUNDS[dtype_name]
        if high is None:
            high = low
            low = 0
        low_arr = _np.asarray(low, dtype=_np.int64)
        high_arr = _np.asarray(high, dtype=_np.int64)
        shape = _np.broadcast_shapes(low_arr.shape, high_arr.shape)
        target = shape if size is None else _shape_of(size)
        try:
            low_b = _np.ascontiguousarray(_np.broadcast_to(low_arr, target), dtype=_np.int64)
            high_b = _np.ascontiguousarray(_np.broadcast_to(high_arr, target), dtype=_np.int64)
        except ValueError:
            raise ValueError("shape mismatch") from None
        if _np.any(low_b < dmin) or _np.any(high_b - 1 > dmax):
            raise ValueError(f"low is out of bounds for {dtype_name}")
        if _np.any(low_b >= high_b):
            raise ValueError("low >= high")
        count = high_b.astype(_np.float64) - low_b.astype(_np.float64)
        values, new_state = _nr._bounded_int_fill(
            self._raw_state(), low_b, count, True, dtype_name
        )
        self._store_state(new_state)
        result = values if dtype_name == "int64" else values.astype(dtype_name)
        return int(_scalar(result)) if target == () else result

    def choice(self, a, size=None, replace=True, p=None):
        pop_size, values_source = _choice_population(a)
        shape = _shape_of(size)
        idx = _choice_indices(
            self._raw_state, self._store_state, pop_size, shape, replace, p, True
        )
        return _choice_result(idx, values_source, size, shape)

    def shuffle(self, x, axis=0):
        return _shuffle(self._raw_state, self._store_state, x, axis=axis)

    def permutation(self, x, axis=0):
        return _permutation(self._raw_state, self._store_state, x, axis=axis)

    def uniform(self, low=0.0, high=1.0, size=None):
        shape, (low_a, high_a) = _broadcast_params(size, (low, high))
        base, new_state = _nr._uniform01_fill(self._raw_state(), shape, "float64")
        self._store_state(new_state)
        result = low_a + (high_a - low_a) * base
        return float(_scalar(result)) if shape == () else result

    def normal(self, loc=0.0, scale=1.0, size=None):
        shape, (loc_a, scale_a) = _broadcast_params(size, (loc, scale))
        if _np.any(scale_a < 0):
            raise ValueError("scale < 0")
        base, new_state = _nr._standard_normal_fill(self._raw_state(), shape, "float64")
        self._store_state(new_state)
        result = loc_a + scale_a * base
        return float(_scalar(result)) if shape == () else result

    def standard_normal(self, size=None):
        shape = _shape_of(size)
        values, new_state = _nr._standard_normal_fill(self._raw_state(), shape, "float64")
        self._store_state(new_state)
        return float(_scalar(values)) if shape == () else values

    def standard_exponential(self, size=None):
        shape = _shape_of(size)
        values, new_state = _nr._standard_exponential_fill(
            self._raw_state(), shape, "float64", "inv"
        )
        self._store_state(new_state)
        return float(_scalar(values)) if shape == () else values

    def exponential(self, scale=1.0, size=None):
        shape, (scale_a,) = _broadcast_params(size, (scale,))
        if _np.any(scale_a < 0):
            raise ValueError("scale < 0")
        base, new_state = _nr._standard_exponential_fill(
            self._raw_state(), shape, "float64", "inv"
        )
        self._store_state(new_state)
        result = scale_a * base
        return float(_scalar(result)) if shape == () else result

    def standard_gamma(self, shape, size=None):
        param_shape, (shape_a,) = _broadcast_params(size, (shape,))
        values, new_state = _nr._standard_gamma_fill(self._raw_state(), shape_a, "float64")
        self._store_state(new_state)
        return float(_scalar(values)) if param_shape == () else values

    def gamma(self, shape, scale=1.0, size=None):
        param_shape, (shape_a, scale_a) = _broadcast_params(size, (shape, scale))
        values, new_state = _nr._standard_gamma_fill(self._raw_state(), shape_a, "float64")
        self._store_state(new_state)
        result = values * scale_a
        return float(_scalar(result)) if param_shape == () else result

    def chisquare(self, df, size=None):
        param_shape, (df_a,) = _broadcast_params(size, (df,))
        values, new_state = _nr._chisquare_fill(self._raw_state(), df_a)
        self._store_state(new_state)
        return float(_scalar(values)) if param_shape == () else values

    def f(self, dfnum, dfden, size=None):
        param_shape, (dfnum_a, dfden_a) = _broadcast_params(size, (dfnum, dfden))
        values, new_state = _nr._f_fill(self._raw_state(), dfnum_a, dfden_a)
        self._store_state(new_state)
        return float(_scalar(values)) if param_shape == () else values

    def standard_t(self, df, size=None):
        param_shape, (df_a,) = _broadcast_params(size, (df,))
        values, new_state = _nr._standard_t_fill(self._raw_state(), df_a)
        self._store_state(new_state)
        return float(_scalar(values)) if param_shape == () else values

    def binomial(self, n, p, size=None):
        n_arr = _np.asarray(n, dtype=_np.int64)
        p_arr = _np.asarray(p, dtype=_np.float64)
        shape = _np.broadcast_shapes(n_arr.shape, p_arr.shape)
        target = shape if size is None else _shape_of(size)
        n_b = _np.ascontiguousarray(_np.broadcast_to(n_arr, target), dtype=_np.int64)
        p_b = _np.ascontiguousarray(_np.broadcast_to(p_arr, target), dtype=_np.float64)
        values, new_state = _nr._binomial_fill(self._raw_state(), n_b, p_b)
        self._store_state(new_state)
        return int(_scalar(values)) if target == () else values

    def poisson(self, lam=1.0, size=None):
        lam_arr = _np.asarray(lam, dtype=_np.float64)
        target = lam_arr.shape if size is None else _shape_of(size)
        lam_b = _np.ascontiguousarray(_np.broadcast_to(lam_arr, target), dtype=_np.float64)
        values, new_state = _nr._poisson_fill(self._raw_state(), lam_b)
        self._store_state(new_state)
        return int(_scalar(values)) if target == () else values


# ---------------------------------------------------------------------------
# Module-level legacy functions, bound to one shared global RandomState.
# ---------------------------------------------------------------------------

_rand = RandomState()


def seed(seed=None):
    _rand.seed(seed)


def get_state(legacy=True):
    return _rand.get_state(legacy)


def set_state(state):
    return _rand.set_state(state)


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
