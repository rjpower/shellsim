"""Exercise ordered insertion, indexing and invalid removal."""

from sortedcontainers import SortedDict, SortedList

values = SortedList([3, 1, 2])
values.add(2)
assert list(values) == [1, 2, 2, 3]
assert values.bisect_left(2) == 1
mapping = SortedDict({"b": 2, "a": 1})
assert list(mapping) == ["a", "b"]
try:
    values.remove(99)
except ValueError:
    pass
else:
    raise AssertionError("absent value removed")
print("sortedcontainers: ordering, indexing and invalid removal passed")
