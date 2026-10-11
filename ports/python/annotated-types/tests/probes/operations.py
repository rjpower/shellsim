"""Exercise a bounded public API in the actual guest."""

from annotated_types import Ge, Len

assert Ge(3).ge == 3
assert Len(min_length=1, max_length=4).max_length == 4
print("annotated-types: guest public API probe passed")
