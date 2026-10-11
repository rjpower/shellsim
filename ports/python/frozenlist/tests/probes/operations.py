"""Exercise the declared local guest API."""

from frozenlist import FrozenList
from frozenlist._frozenlist import FrozenList as NativeList

assert FrozenList is NativeList
values = FrozenList([1, 2])
values.append(3)
assert list(values) == [1, 2, 3]
values.freeze()
assert hash(values) == hash((1, 2, 3))
try:
    values.append(4)
except RuntimeError:
    pass
else:
    raise AssertionError("frozen list was mutable")
print("frozenlist: local guest operations passed")
