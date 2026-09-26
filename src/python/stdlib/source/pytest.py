"""Small pytest facade used by shellsim's source-driven runner."""

import re

from _pytest import fail, skip


class RaisesContext:
    """Context manager returned by ``raises``; ``type`` and ``value`` describe the caught error."""

    def __init__(self, expected, match):
        self.expected = expected
        self.match = match
        self.type = None
        self.value = None

    def __enter__(self):
        return self

    def __exit__(self, kind, value, traceback):
        if kind is None:
            fail(f"DID NOT RAISE {self.expected}")
        if not issubclass(kind, self.expected):
            return False
        if self.match is not None and re.search(self.match, str(value)) is None:
            fail(f"Regex pattern did not match.\n Regex: {self.match!r}\n Input: {str(value)!r}")
        self.type = kind
        self.value = value
        return True


def raises(expected, *, match=None):
    return RaisesContext(expected, match)


def fixture(function=None, scope=None, params=None, autouse=False, ids=None, name=None):
    if function is not None:
        return function

    def decorate(candidate):
        return candidate

    return decorate


class _Mark:
    def parametrize(self, argnames, argvalues, ids=None):
        def decorate(candidate):
            return candidate

        return decorate

    def skip(self, reason=None):
        def decorate(candidate):
            return candidate

        return decorate

    def skipif(self, condition, reason=None):
        def decorate(candidate):
            return candidate

        return decorate


mark = _Mark()
