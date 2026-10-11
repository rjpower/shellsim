"""Exercise a bounded public API inside the actual guest."""

from more_itertools import chunked, windowed

assert list(chunked(range(5), 2)) == [[0, 1], [2, 3], [4]]
assert list(windowed([1, 2, 3], 2)) == [(1, 2), (2, 3)]
print("more-itertools: public API probe passed")
