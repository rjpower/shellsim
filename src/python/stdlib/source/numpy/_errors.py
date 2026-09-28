"""Floating-point error reporting (``errstate``, ``seterr``/``geterr``) and dtype limits
(``finfo``/``iinfo``).

Ufunc loops record raised IEEE-style flags natively (see ``numpy/ops.rs``) without touching
Python. Once a loop finishes, the interpreter calls ``_report`` with the flags it saw; this module
holds the mode table those flags are checked against, so ``errstate`` nests and restores like any
context manager and native code pays nothing when no flag is raised. ``finfo``/``iinfo`` live
alongside it since both describe the numeric boundaries a dtype's arithmetic runs into.
"""

import math
import warnings

import numpy as np
from numpy.exceptions import ComplexWarning

__all__ = ["errstate", "finfo", "geterr", "geterrcall", "iinfo", "seterr", "seterrcall"]

_CATEGORIES = ("divide", "over", "under", "invalid")
_MODES = frozenset({"ignore", "warn", "raise", "call", "print", "log"})
_MESSAGES = {
    "divide": "divide by zero",
    "over": "overflow",
    "under": "underflow",
    "invalid": "invalid value",
}

# The default state a fresh interpreter starts with: NumPy warns on everything but underflow,
# which is common and rarely actionable.
_state = {"divide": "warn", "over": "warn", "under": "ignore", "invalid": "warn"}
_callback = None


def _validate(mode):
    if mode not in _MODES:
        raise ValueError(f"invalid error mode {mode!r}")


def seterr(all=None, divide=None, over=None, under=None, invalid=None):
    """Set how floating-point errors are handled; return the previous settings."""
    previous = dict(_state)
    requested = {"divide": divide, "over": over, "under": under, "invalid": invalid}
    for category, mode in requested.items():
        chosen = all if mode is None else mode
        if chosen is not None:
            _validate(chosen)
    for category, mode in requested.items():
        chosen = all if mode is None else mode
        if chosen is not None:
            _state[category] = chosen
    return previous


def geterr():
    """The current floating-point error mode for each category."""
    return dict(_state)


def seterrcall(func):
    """Install the callable used by ``call`` mode; return the previous one."""
    global _callback
    if func is not None and not callable(func) and not hasattr(func, "write"):
        raise ValueError("Only callable can be used as callback")
    previous = _callback
    _callback = func
    return previous


def geterrcall():
    """The callable installed by ``seterrcall``, or ``None``."""
    return _callback


class errstate:
    """Context manager (and decorator) that overrides floating-point error modes.

    ``np.errstate(divide="raise")`` raises inside the block; other categories keep their current
    mode unless ``all=`` or their own keyword is also given. Settings restore on exit even when
    the block raises.
    """

    def __init__(self, *, all=None, divide=None, over=None, under=None, invalid=None):
        self._all = all
        self._requested = {"divide": divide, "over": over, "under": under, "invalid": invalid}
        for mode in self._requested.values():
            if mode is not None:
                _validate(mode)
        if all is not None:
            _validate(all)
        self._previous = None

    def __enter__(self):
        self._previous = dict(_state)
        for category, mode in self._requested.items():
            chosen = self._all if mode is None else mode
            if chosen is not None:
                _state[category] = chosen
        return self

    def __exit__(self, *exc_info):
        _state.update(self._previous)
        return False

    def __call__(self, function):
        def wrapped(*args, **kwargs):
            with errstate(
                all=self._all,
                divide=self._requested["divide"],
                over=self._requested["over"],
                under=self._requested["under"],
                invalid=self._requested["invalid"],
            ):
                return function(*args, **kwargs)

        return wrapped


def _act(category, name):
    """Apply the current mode for `category` to a flag raised by ufunc `name`."""
    mode = _state[category]
    if mode == "ignore":
        return
    message = f"{_MESSAGES[category]} encountered in {name}"
    if mode == "warn":
        warnings.warn(message, RuntimeWarning, stacklevel=3)
    elif mode == "raise":
        raise FloatingPointError(message)
    elif mode == "call":
        if _callback is not None:
            flag = 1 << _CATEGORIES.index(category)
            _callback(_MESSAGES[category], flag)
    elif mode in ("print", "log"):
        print(f"Warning: {message}")


def _report(name, divide, overflow, underflow, invalid):
    """Apply the current error modes to the flags raised by one ufunc call.

    Categories are checked in a fixed order (divide, overflow, underflow, invalid) so that a
    ``raise`` mode reports the first flag in that order and a ``warn`` mode for an earlier
    category is not skipped by a later ``raise``.
    """
    if divide:
        _act("divide", name)
    if overflow:
        _act("over", name)
    if underflow:
        _act("under", name)
    if invalid:
        _act("invalid", name)


def _warn_complex_discard():
    warnings.warn(
        "Casting complex values to real discards the imaginary part", ComplexWarning, stacklevel=2
    )


def _warn_where_without_out():
    warnings.warn(
        "'where' used without 'out', expect uninitialized memory in output. "
        "If this is intentional, use out=None.",
        UserWarning,
        stacklevel=2,
    )


# -- dtype limits ---------------------------------------------------------------------------
#
# Every value below follows directly from the IEEE 754 layout of a float dtype (mantissa and
# exponent bit counts) or from the bit width of an integer dtype, computed with the standard
# formulas rather than a table of literals copied from any implementation.

import numpy as np  # noqa: E402  (kept separate from the errstate imports above for clarity)

# Mantissa and exponent bit counts (excluding the sign bit and, for the mantissa, the implicit
# leading one) for the three floating dtypes shellsim models.
_LAYOUT = {
    "float16": (10, 5),
    "float32": (23, 8),
    "float64": (52, 11),
}
# A complex dtype's numeric limits are those of its real and imaginary components.
_COMPLEX_TO_FLOAT = {"complex64": "float32", "complex128": "float64"}


def _resolve_dtype(dtype_like):
    """Resolve any of `finfo`/`iinfo`'s accepted spellings to a ``numpy.dtype``.

    NumPy accepts a dtype, a scalar type, a dtype-spec string, or a value whose *type* names a
    dtype (``finfo(3)`` uses ``dtype(int)``, matching NumPy).
    """
    try:
        return np.dtype(dtype_like)
    except TypeError:
        return np.dtype(type(dtype_like))


class finfo:
    """Machine limits for a floating or complex floating dtype.

    Example
    -------
    >>> np.finfo(np.float32).eps
    1.1920929e-07
    """

    def __init__(self, dtype):
        resolved = _resolve_dtype(dtype)
        float_name = _COMPLEX_TO_FLOAT.get(resolved.name, resolved.name)
        if float_name not in _LAYOUT:
            raise ValueError(f"data type {resolved!r} not compatible with finfo")
        nmant, nexp = _LAYOUT[float_name]
        self.dtype = np.dtype(float_name)
        self.bits = self.dtype.itemsize * 8
        precision_bits = nmant + 1
        self.nmant = nmant
        self.nexp = nexp
        self.iexp = nexp
        self.machep = -nmant
        self.negep = -(nmant + 1)
        self.maxexp = 2 ** (nexp - 1)
        self.minexp = -self.maxexp + 2
        self.eps = self.dtype.type(2.0**self.machep)
        self.epsneg = self.dtype.type(2.0**self.negep)
        self.max = self.dtype.type((2.0 - 2.0 ** (1 - precision_bits)) * 2.0 ** (self.maxexp - 1))
        self.min = self.dtype.type(-self.max)
        self.smallest_normal = self.dtype.type(2.0**self.minexp)
        self.tiny = self.smallest_normal
        self.smallest_subnormal = self.dtype.type(2.0 ** (self.minexp - nmant))
        self.precision = int(nmant * math.log10(2))
        self.resolution = self.dtype.type(10.0**-self.precision)

    def __repr__(self):
        return f"finfo(resolution={self.resolution!r}, min={self.min!r}, max={self.max!r}, dtype={self.dtype})"


class iinfo:
    """Machine limits for an integer dtype.

    Example
    -------
    >>> np.iinfo(np.int8).max
    127
    """

    def __init__(self, dtype):
        resolved = _resolve_dtype(dtype)
        if resolved.kind not in "iu":
            raise ValueError(f"Invalid integer data type {resolved.char!r}.")
        self.dtype = resolved
        self.bits = resolved.itemsize * 8
        self.kind = resolved.kind
        if resolved.kind == "u":
            self.min = 0
            self.max = 2**self.bits - 1
        else:
            self.min = -(2 ** (self.bits - 1))
            self.max = 2 ** (self.bits - 1) - 1

    def __repr__(self):
        return f"iinfo(min={self.min}, max={self.max}, dtype={self.dtype})"
