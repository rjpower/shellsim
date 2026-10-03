"""Exception groups (PEP 654) as ordinary exception classes.

``BaseExceptionGroup`` and ``ExceptionGroup`` are exposed as builtins from here. Only the class
surface is modeled: ``except*`` clauses are not supported by the parser, and exception classes
are constructed without ``__new__``, so ``BaseExceptionGroup(...)`` keeps that class instead of
becoming an ``ExceptionGroup`` when every member is an ``Exception``.
"""


class BaseExceptionGroup(BaseException):
    """A group of exceptions raised together."""

    def __init__(self, message, exceptions):
        if not isinstance(message, str):
            raise TypeError("argument 1 must be str, not " + type(message).__name__)
        exceptions = tuple(exceptions)
        if not exceptions:
            raise ValueError("second argument (exceptions) must be a non-empty sequence")
        for index, item in enumerate(exceptions):
            if not isinstance(item, BaseException):
                raise ValueError(
                    "Item %d of second argument (exceptions) is not an exception" % index
                )
        if isinstance(self, Exception) and not all(isinstance(item, Exception) for item in exceptions):
            raise TypeError("Cannot nest BaseExceptions in an ExceptionGroup")
        BaseException.__init__(self, message, exceptions)
        self.message = message
        self.exceptions = exceptions

    def __str__(self):
        count = len(self.exceptions)
        return "%s (%d sub-exception%s)" % (self.message, count, "" if count == 1 else "s")

    def __repr__(self):
        return "%s(%r, %r)" % (type(self).__name__, self.message, list(self.exceptions))

    def derive(self, excs):
        """A new group with the same message holding ``excs``; subclasses override this."""
        return BaseExceptionGroup(self.message, excs)

    def _derive_like(self, excs):
        group = self.derive(excs)
        notes = getattr(self, "__notes__", None)
        if notes is not None:
            group.__notes__ = list(notes)
        return group

    def _matcher(self, condition):
        if isinstance(condition, type) and issubclass(condition, BaseException):
            return lambda exc: isinstance(exc, condition)
        if isinstance(condition, tuple) and all(
            isinstance(item, type) and issubclass(item, BaseException) for item in condition
        ):
            return lambda exc: isinstance(exc, condition)
        if callable(condition):
            return condition
        raise TypeError("expected an exception type, a tuple of exception types, or a callable")

    def split(self, condition):
        """``(match, rest)``: the sub-tree of exceptions matching ``condition`` and the remainder."""
        matches = self._matcher(condition)
        if matches(self):
            return self, None
        matched = []
        rest = []
        for exc in self.exceptions:
            if isinstance(exc, BaseExceptionGroup):
                inner_match, inner_rest = exc.split(condition)
                if inner_match is not None:
                    matched.append(inner_match)
                if inner_rest is not None:
                    rest.append(inner_rest)
            elif matches(exc):
                matched.append(exc)
            else:
                rest.append(exc)
        return (
            self._derive_like(matched) if matched else None,
            self._derive_like(rest) if rest else None,
        )

    def subgroup(self, condition):
        return self.split(condition)[0]


class ExceptionGroup(BaseExceptionGroup, Exception):
    """A group whose members are all ``Exception`` instances."""

    def derive(self, excs):
        return ExceptionGroup(self.message, excs)
