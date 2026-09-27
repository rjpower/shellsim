"""Small pytest facade used by shellsim's source-driven runner."""

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
