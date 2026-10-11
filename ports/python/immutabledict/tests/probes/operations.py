"""Exercise a bounded public API in the actual guest."""

from immutabledict import immutabledict

value = immutabledict({"a": 1})
assert value["a"] == 1
assert hash(value) == hash(immutabledict({"a": 1}))
try:
    value["a"] = 2
except TypeError:
    pass
else:
    raise AssertionError("immutable mapping changed")
print("immutabledict: guest public API probe passed")
