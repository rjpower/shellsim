"""Floating-point error handling for shellsim's NumPy: ``errstate``, ``seterr`` and ``geterr``.

Ufunc loops record divide-by-zero, overflow, underflow and invalid-operation flags natively and
call ``_report`` after the loop when any flag is set. The modes kept here decide whether each
flag is ignored, reported as a ``RuntimeWarning``, or raised as ``FloatingPointError``.
"""

import warnings

from _numpy import _ComplexWarning as ComplexWarning

_MODES = ("ignore", "warn", "raise", "call", "print", "log")
_CATEGORIES = ("divide", "over", "under", "invalid")
_MESSAGES = {
    "divide": "divide by zero",
    "over": "overflow",
    "under": "underflow",
    "invalid": "invalid value",
}
_state = {"divide": "warn", "over": "warn", "under": "ignore", "invalid": "warn"}


def geterr():
    """Return the current mode for each floating-point error category."""
    return dict(_state)


def _updates(all=None, divide=None, over=None, under=None, invalid=None):
    updates = {}
    if all is not None:
        for category in _CATEGORIES:
            updates[category] = all
    for category, mode in (("divide", divide), ("over", over), ("under", under), ("invalid", invalid)):
        if mode is not None:
            updates[category] = mode
    for mode in updates.values():
        if mode not in _MODES:
            raise ValueError(f"invalid error mode {mode!r}; valid modes are {list(_MODES)}")
        if mode in ("call", "print", "log"):
            raise NotImplementedError(f"numpy error mode {mode!r} is not supported by shellsim")
    return updates


def seterr(all=None, divide=None, over=None, under=None, invalid=None):
    """Set the error modes and return the previous ones."""
    updates = _updates(all, divide, over, under, invalid)
    previous = geterr()
    _state.update(updates)
    return previous


class errstate:
    """Context manager and decorator that sets error modes for a block and restores them."""

    def __init__(self, *, call=None, all=None, divide=None, over=None, under=None, invalid=None):
        if call is not None:
            raise NotImplementedError("numpy.errstate(call=...) is not supported by shellsim")
        self._updates = _updates(all, divide, over, under, invalid)
        self._saved = []

    def __enter__(self):
        self._saved.append(geterr())
        _state.update(self._updates)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        _state.clear()
        _state.update(self._saved.pop())
        return False

    def __call__(self, function):
        def wrapper(*args, **kwargs):
            with self:
                return function(*args, **kwargs)

        return wrapper


def _report(name, divide, over, under, invalid):
    """Apply the error modes to the flags one ufunc call raised, in NumPy's order."""
    for category, raised in (("divide", divide), ("over", over), ("under", under), ("invalid", invalid)):
        if not raised:
            continue
        mode = _state[category]
        if mode == "ignore":
            continue
        message = f"{_MESSAGES[category]} encountered in {name}"
        if mode == "raise":
            raise FloatingPointError(message)
        warnings.warn(message, RuntimeWarning, stacklevel=2)


def _warn_complex_discard():
    """Warn that a conversion to a real type dropped an imaginary part."""
    warnings.warn(
        "Casting complex values to real discards the imaginary part", ComplexWarning, stacklevel=2
    )
