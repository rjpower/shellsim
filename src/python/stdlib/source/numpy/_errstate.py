"""Floating-point error handling for shellsim's NumPy: ``errstate``, ``seterr``, ``geterr``,
``seterrcall`` and ``geterrcall``.

Ufunc loops record divide-by-zero, overflow, underflow and invalid-operation flags natively and
call ``_report`` after the loop when any flag is set. The modes kept here decide whether each
flag is ignored, warned about, raised, printed, passed to a callback, or written to a log
object, following ``numpy/_core/src/umath/extobj.c``.
"""

import os
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
# The status bits NumPy passes to a 'call' handler.
_BITS = {"divide": 1, "over": 2, "under": 4, "invalid": 8}

_state = {"divide": "warn", "over": "warn", "under": "ignore", "invalid": "warn"}
_handler = None
_Unspecified = object()


def geterr():
    """Return the current mode for each floating-point error category."""
    return dict(_state)


def geterrcall():
    """Return the callback or log object used by the 'call' and 'log' modes."""
    return _handler


def _configuration(call, all, divide, over, under, invalid):
    """Validate a change and return the modes and handler it produces, as ``_make_extobj``."""
    modes = dict(_state)
    for category, mode in (("all", all), ("divide", divide), ("over", over), ("under", under),
                           ("invalid", invalid)):
        if mode is None:
            continue
        if mode not in _MODES:
            raise ValueError(f"invalid error mode {mode!r}")
        if category == "all":
            for name in _CATEGORIES:
                modes[name] = mode
        else:
            modes[category] = mode
    handler = _handler
    if call is not _Unspecified:
        if call is not None and not callable(call):
            write = getattr(call, "write", None)
            if write is None or not callable(write):
                raise TypeError("python object must be callable or have a callable write method")
        handler = call
    return modes, handler


def _install(modes, handler):
    global _handler
    _state.clear()
    _state.update(modes)
    _handler = handler


def seterr(all=None, divide=None, over=None, under=None, invalid=None):
    """Set the error modes and return the previous ones."""
    previous = geterr()
    _install(*_configuration(_Unspecified, all, divide, over, under, invalid))
    return previous


def seterrcall(func):
    """Set the 'call' mode callback or 'log' mode object and return the previous one."""
    previous = geterrcall()
    _install(*_configuration(func, None, None, None, None, None))
    return previous


class errstate:
    """Context manager and decorator that sets error modes for a block and restores them."""

    def __init__(self, *, call=_Unspecified, all=None, divide=None, over=None, under=None,
                 invalid=None):
        self._token = None
        self._call = call
        self._settings = (all, divide, over, under, invalid)

    def __enter__(self):
        if self._token is not None:
            raise TypeError("Cannot enter `np.errstate` twice.")
        configuration = _configuration(self._call, *self._settings)
        self._token = (geterr(), _handler)
        _install(*configuration)

    def __exit__(self, *exc_info):
        _install(*self._token)
        self._token = None

    def __call__(self, func):
        def inner(*args, **kwargs):
            configuration = _configuration(self._call, *self._settings)
            saved = (geterr(), _handler)
            _install(*configuration)
            try:
                return func(*args, **kwargs)
            finally:
                _install(*saved)

        inner.__name__ = getattr(func, "__name__", "inner")
        inner.__doc__ = getattr(func, "__doc__", None)
        return inner


def _report(name, divide, over, under, invalid):
    """Apply the error modes to the flags one ufunc call raised, in NumPy's order."""
    raised = {"divide": divide, "over": over, "under": under, "invalid": invalid}
    status = sum(_BITS[category] for category in _CATEGORIES if raised[category])
    for category in _CATEGORIES:
        if not raised[category]:
            continue
        mode = _state[category]
        errtype = _MESSAGES[category]
        if mode == "ignore":
            continue
        if mode == "warn":
            warnings.warn(f"{errtype} encountered in {name}", RuntimeWarning, stacklevel=2)
        elif mode == "raise":
            raise FloatingPointError(f"{errtype} encountered in {name}")
        elif mode == "print":
            os.write(2, f"Warning: {errtype} encountered in {name}\n".encode())
        elif mode == "call":
            if _handler is None:
                raise NameError(
                    f"python callback specified for {errtype} (in  {name}) but no function found."
                )
            _handler(errtype, status)
        elif _handler is None:
            raise NameError(
                f"log specified for {errtype} (in {name}) but no object with write method found."
            )
        else:
            _handler.write(f"Warning: {errtype} encountered in {name}\n")


def _warn_complex_discard():
    """Warn that a conversion to a real type dropped an imaginary part."""
    warnings.warn(
        "Casting complex values to real discards the imaginary part", ComplexWarning, stacklevel=2
    )


def _warn_where_without_out():
    """Warn that a ufunc's ``where=`` mask without ``out=`` leaves elements uninitialized."""
    warnings.warn(
        "'where' used without 'out', expect uninitialized memory in output. If this is "
        "intentional, use out=None.",
        UserWarning,
        stacklevel=2,
    )
