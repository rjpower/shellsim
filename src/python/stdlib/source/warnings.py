"""Warning control, following CPython 3.14's ``warnings`` module.

Filters, actions, ``catch_warnings`` and the display format follow CPython. A warning's location
comes from the simulated call stack. As in tracebacks, every frame reports the script path, and
the module name that ``module=`` filters match is always ``__main__``.
"""

import re
import sys

from _shellsim_warnings import caller as _caller

__all__ = [
    "warn",
    "warn_explicit",
    "showwarning",
    "formatwarning",
    "filterwarnings",
    "simplefilter",
    "resetwarnings",
    "catch_warnings",
]

_ACTIONS = ("error", "ignore", "always", "all", "default", "module", "once")

defaultaction = "default"
filters = [
    ("default", None, DeprecationWarning, re.compile("__main__"), 0),
    ("ignore", None, DeprecationWarning, None, 0),
    ("ignore", None, PendingDeprecationWarning, None, 0),
    ("ignore", None, ImportWarning, None, 0),
    ("ignore", None, ResourceWarning, None, 0),
]

# CPython keeps "default" and "module" state per module in ``__warningregistry__``; one registry
# keyed by filename is equivalent here because every frame reports the same script path.
_registry = {}
_once_registry = {}


class WarningMessage:
    """One emitted warning, as recorded by ``catch_warnings(record=True)``."""

    _WARNING_DETAILS = ("message", "category", "filename", "lineno", "file", "line", "source")

    def __init__(self, message, category, filename, lineno, file=None, line=None, source=None):
        self.message = message
        self.category = category
        self.filename = filename
        self.lineno = lineno
        self.file = file
        self.line = line
        self.source = source
        self._category_name = category.__name__ if category else None

    def __str__(self):
        return (
            f"{{message : {self.message!r}, category : {self._category_name!r}, "
            f"filename : {self.filename!r}, lineno : {self.lineno}, line : {self.line!r}}}"
        )


def _source_line(filename, lineno):
    try:
        with open(filename) as handle:
            for number, text in enumerate(handle, 1):
                if number == lineno:
                    return text
    except (OSError, UnicodeDecodeError):
        return None
    return None


def formatwarning(message, category, filename, lineno, line=None):
    """Return the text ``showwarning`` writes for one warning."""
    text = f"{filename}:{lineno}: {category.__name__}: {message}\n"
    if line is None:
        line = _source_line(filename, lineno)
    if line:
        text += f"  {line.strip()}\n"
    return text


def _showwarnmsg_impl(msg):
    file = msg.file if msg.file is not None else sys.stderr
    text = formatwarning(msg.message, msg.category, msg.filename, msg.lineno, msg.line)
    try:
        file.write(text)
    except OSError:
        pass


def showwarning(message, category, filename, lineno, file=None, line=None):
    """Write a warning to ``file``, or to ``sys.stderr``."""
    _showwarnmsg_impl(WarningMessage(message, category, filename, lineno, file, line))


_showwarning_orig = showwarning


def _showwarnmsg(msg):
    if showwarning is not _showwarning_orig:
        if not callable(showwarning):
            raise TypeError("warnings.showwarning() must be set to a function or method")
        showwarning(msg.message, msg.category, msg.filename, msg.lineno, msg.file, msg.line)
        return
    _showwarnmsg_impl(msg)


def _filters_mutated():
    global _registry, _once_registry
    _registry = {}
    _once_registry = {}


def _check_category(category):
    if not (isinstance(category, type) and issubclass(category, Warning)):
        raise TypeError("category must be a Warning subclass")


def _check_filter(action, lineno):
    if action not in _ACTIONS:
        raise ValueError(f"invalid action: {action!r}")
    if not isinstance(lineno, int) or lineno < 0:
        raise ValueError("lineno must be an int >= 0")


def _add_filter(item, append):
    global filters
    if not append:
        if item in filters:
            filters.remove(item)
        filters.insert(0, item)
    elif item not in filters:
        filters.append(item)
    _filters_mutated()


def filterwarnings(action, message="", category=Warning, module="", lineno=0, append=False):
    """Insert a filter matching a message regex, category, module regex and line."""
    _check_filter(action, lineno)
    if not isinstance(message, str):
        raise TypeError("message must be a string")
    if not isinstance(module, str):
        raise TypeError("module must be a string")
    _check_category(category)
    message_pattern = re.compile(message, re.I) if message else None
    module_pattern = re.compile(module) if module else None
    _add_filter((action, message_pattern, category, module_pattern, lineno), append)


def simplefilter(action, category=Warning, lineno=0, append=False):
    """Insert a filter that matches every message and module."""
    _check_filter(action, lineno)
    _check_category(category)
    _add_filter((action, None, category, None, lineno), append)


def resetwarnings():
    """Remove every filter, including the defaults."""
    filters[:] = []
    _filters_mutated()


def _matching_action(text, category, module, lineno):
    for action, message, filter_category, filter_module, filter_lineno in filters:
        if (
            (message is None or message.match(text))
            and issubclass(category, filter_category)
            and (filter_module is None or filter_module.match(module))
            and (filter_lineno == 0 or lineno == filter_lineno)
        ):
            return action
    return defaultaction


def warn(message, category=None, stacklevel=1, source=None, *, skip_file_prefixes=()):
    """Issue a warning attributed to the caller ``stacklevel`` frames up."""
    if isinstance(message, Warning):
        category = type(message)
    if category is None:
        category = UserWarning
    if not (isinstance(category, type) and issubclass(category, Warning)):
        raise TypeError(f"category must be a Warning subclass, not '{type(category).__name__}'")
    location = _caller(stacklevel)
    filename, lineno = ("sys", 1) if location is None else location
    warn_explicit(message, category, filename, lineno, "__main__", source=source)


def warn_explicit(
    message,
    category,
    filename,
    lineno,
    module=None,
    registry=None,
    module_globals=None,
    source=None,
):
    """Issue a warning at an explicit location."""
    lineno = int(lineno)
    if module is None:
        module = filename or "<unknown>"
        if module[-3:].lower() == ".py":
            module = module[:-3]
    if isinstance(message, Warning):
        text = str(message)
        category = type(message)
    else:
        text = message
        message = category(message)
    action = _matching_action(text, category, module, lineno)
    if action not in _ACTIONS:
        raise RuntimeError(f"Unrecognized action ({action!r}) in warnings.filters")
    if action == "ignore":
        return
    if action == "error":
        raise message
    if action == "once":
        key = (text, category)
        if key in _once_registry:
            return
        _once_registry[key] = True
    elif action == "module":
        key = (text, category, filename, 0)
        if key in _registry:
            return
        _registry[key] = True
    elif action == "default":
        key = (text, category, filename, lineno)
        if key in _registry:
            return
        _registry[key] = True
    _showwarnmsg(WarningMessage(message, category, filename, lineno, source=source))


class catch_warnings:
    """Save and restore the warning filters and display hooks around a block.

    With ``record=True`` the block receives a list that collects each shown warning as a
    ``WarningMessage`` instead of writing it.
    """

    def __init__(
        self,
        *,
        record=False,
        module=None,
        action=None,
        category=Warning,
        lineno=0,
        append=False,
    ):
        self._record = record
        self._filter = None if action is None else (action, category, lineno, append)
        self._entered = False

    def __repr__(self):
        return f"catch_warnings(record={self._record!r})"

    def __enter__(self):
        global filters, showwarning, _showwarnmsg_impl
        if self._entered:
            raise RuntimeError(f"Cannot enter {self!r} twice")
        self._entered = True
        self._saved = (filters, showwarning, _showwarnmsg_impl)
        filters = filters[:]
        _filters_mutated()
        if self._filter is not None:
            simplefilter(*self._filter)
        if not self._record:
            return None
        log = []
        _showwarnmsg_impl = log.append
        showwarning = _showwarning_orig
        return log

    def __exit__(self, *exc_info):
        global filters, showwarning, _showwarnmsg_impl
        if not self._entered:
            raise RuntimeError(f"Cannot exit {self!r} without entering first")
        filters, showwarning, _showwarnmsg_impl = self._saved
        _filters_mutated()
        return False
