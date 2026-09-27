"""Small pytest facade used by shellsim's source-driven runner."""

import math
import re
import warnings

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


class WarningsChecker:
    """Context manager returned by ``warns``; ``list`` holds every warning the block emitted."""

    def __init__(self, expected, match):
        self.expected = expected if isinstance(expected, tuple) else (expected,)
        self.match = match
        self.list = []
        self._catcher = warnings.catch_warnings(record=True)

    def __enter__(self):
        self.list = self._catcher.__enter__()
        warnings.simplefilter("always")
        return self

    def __exit__(self, kind, value, traceback):
        self._catcher.__exit__(kind, value, traceback)
        if kind is not None:
            return False
        for record in self.list:
            if issubclass(record.category, self.expected) and (
                self.match is None or re.search(self.match, str(record.message)) is not None
            ):
                return False
        matching = "" if self.match is None else f" matching the regex {self.match!r}"
        emitted = [record.category(str(record.message)) for record in self.list]
        fail(
            f"DID NOT WARN. No warnings of type {self.expected!r}{matching} were emitted.\n"
            f" Emitted warnings: {emitted!r}."
        )


def warns(expected_warning=Warning, *, match=None):
    return WarningsChecker(expected_warning, match)


class ApproxBase:
    """The object ``approx`` returns: equal to numbers within a tolerance of ``expected``, and
    to lists, tuples, dicts and NumPy arrays whose numbers all are.

    The tolerance is the larger of ``rel * abs(expected)`` (``rel`` defaults to 1e-6) and
    ``abs`` (default 1e-12); giving ``abs`` alone drops the relative part. NaN equals NaN only
    with ``nan_ok``. A NumPy array compared with a scalar must match it in every element.
    ``__array_ufunc__ = None`` makes arrays defer their ``==`` to this class.
    """

    __array_ufunc__ = None
    __array_priority__ = 100
    __hash__ = None

    def __init__(self, expected, rel=None, abs=None, nan_ok=False):  # noqa: A002
        self.expected = expected
        self.rel = rel
        self.abs = abs
        self.nan_ok = nan_ok

    def __eq__(self, actual):
        expected = _plain(self.expected)
        if _is_array(actual):
            if not isinstance(expected, (list, tuple, dict)):
                return all(self._equal(item, expected) for item in actual.ravel().tolist())
            actual = actual.tolist()
        return self._equal(actual, expected)

    def __ne__(self, actual):
        return not self == actual

    def __repr__(self):
        expected = _plain(self.expected)
        if isinstance(expected, dict):
            items = ", ".join(f"{key!r}: {self._repr_one(value)}" for key, value in expected.items())
            return f"approx({{{items}}})"
        if isinstance(expected, (list, tuple)):
            items = ", ".join(self._repr_one(value) for value in expected)
            return f"approx([{items}])" if isinstance(expected, list) else f"approx(({items}))"
        return self._repr_one(expected)

    def _equal(self, actual, expected):
        if isinstance(expected, dict):
            return (
                isinstance(actual, dict)
                and actual.keys() == expected.keys()
                and all(self._equal(actual[key], value) for key, value in expected.items())
            )
        if isinstance(expected, (list, tuple)):
            return (
                isinstance(actual, (list, tuple))
                and len(actual) == len(expected)
                and all(self._equal(a, e) for a, e in zip(actual, expected))
            )
        if actual == expected:
            return True
        try:
            difference = abs(expected - actual)
        except TypeError:
            return False
        if expected != expected or actual != actual:
            return self.nan_ok and expected != expected and actual != actual
        if abs(expected) == math.inf:
            return False
        return difference <= self._tolerance(expected)

    def _tolerance(self, expected):
        absolute = 1e-12 if self.abs is None else self.abs
        if absolute < 0:
            raise ValueError(f"absolute tolerance can't be negative: {absolute}")
        if self.rel is None and self.abs is not None:
            return absolute
        relative = 1e-6 if self.rel is None else self.rel
        if relative < 0:
            raise ValueError(f"relative tolerance can't be negative: {relative}")
        return max(relative * abs(expected), absolute)

    def _repr_one(self, expected):
        if isinstance(expected, bool) or not isinstance(expected, (int, float, complex)):
            return str(expected)
        if expected != expected:
            return "nan ± ???"
        if abs(expected) == math.inf:
            return str(expected)
        tolerance = f"{expected} ± {self._tolerance(expected):.1e}"
        return f"{tolerance} ∠ ±180°" if isinstance(expected, complex) else tolerance


def _is_array(value):
    # NumPy is frozen into shellsim, so importing it here always succeeds and costs one import.
    import numpy

    return isinstance(value, numpy.ndarray)


def _plain(value):
    """`value` with a NumPy array replaced by its nested lists (or scalar, when 0-d)."""
    return value.tolist() if _is_array(value) else value


def approx(expected, rel=None, abs=None, nan_ok=False):  # noqa: A002
    """Compare numbers, or containers of them, within a tolerance; see ``ApproxBase``."""
    if isinstance(expected, (list, tuple)):
        for index, item in enumerate(expected):
            if isinstance(item, (list, tuple, dict)):
                raise TypeError(
                    f"pytest.approx() does not support nested data structures: {item!r} at index "
                    f"{index}\n  full sequence: {expected!r}"
                )
    return ApproxBase(expected, rel, abs, nan_ok)


def fixture(function=None, scope=None, params=None, autouse=False, ids=None, name=None):
    if function is not None:
        return function

    def decorate(candidate):
        return candidate

    return decorate


class _Mark:
    def parametrize(self, argnames, argvalues, ids=None):
        """Record the evaluated rows on the test function for the runner to call it with."""
        if isinstance(argnames, str):
            names = [name.strip() for name in argnames.split(",") if name.strip()]
        else:
            names = list(argnames)
        rows = []
        for row in argvalues:
            values = (row,) if len(names) == 1 else tuple(row)
            if len(values) != len(names):
                raise ValueError(
                    f'in "parametrize" the number of names ({len(names)}): {tuple(names)} must be '
                    f"equal to the number of values ({len(values)}): {values}"
                )
            rows.append(dict(zip(names, values)))

        def decorate(candidate):
            candidate._shellsim_parametrize = [*getattr(candidate, "_shellsim_parametrize", []), rows]
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


def _parametrized_cases(function):
    """The keyword arguments of each call the runner makes to ``function``.

    Decorators apply bottom-up, so the bottom decorator's rows are recorded first. As in pytest,
    the top decorator's values vary fastest.
    """
    cases = [{}]
    for rows in getattr(function, "_shellsim_parametrize", []):
        cases = [{**case, **row} for case in cases for row in rows]
    return cases
