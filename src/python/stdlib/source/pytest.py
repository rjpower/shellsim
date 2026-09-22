"""Small pytest facade used by shellsim's source-driven runner."""

from _pytest import fail, raises, skip


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
