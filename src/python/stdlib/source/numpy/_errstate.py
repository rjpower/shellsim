"""Floating-point error reporting: ``errstate``, ``seterr``/``geterr``, and their call site.

Ufunc loops record raised IEEE-style flags natively (see ``numpy/ops.rs``) without touching
Python. Once a loop finishes, the interpreter calls ``_report`` with the flags it saw; this module
holds the mode table those flags are checked against, so ``errstate`` nests and restores like any
context manager and native code pays nothing when no flag is raised.
"""

import warnings

from numpy.exceptions import ComplexWarning

__all__ = ["errstate", "seterr", "geterr", "seterrcall", "geterrcall"]

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
