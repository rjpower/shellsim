"""Exercise the declared local guest API."""

from propcache._helpers_c import cached_property as NativeProperty
from propcache.api import cached_property

assert cached_property is NativeProperty


class Value:
    def __init__(self):
        self.calls = 0

    @cached_property
    def answer(self):
        self.calls += 1
        return 42


value = Value()
assert value.answer == value.answer == 42
assert value.calls == 1
del value.answer
assert value.answer == 42 and value.calls == 2
print("propcache: local guest operations passed")
